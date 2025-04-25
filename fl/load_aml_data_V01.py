import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_aml_data(data_path: Path, batch_size: int = 1024, bank_filter: int | None = None):
    # Load dataset
    df = pd.read_csv(data_path)

    # Business logic: remove formats with no suspicious activity
    df = df[~df['Payment Format'].isin(['Reinvestment', 'Wire'])]

    # New feature columns
    df['currency_change'] = np.where(df['Receiving Currency'] != df['Payment Currency'], 1, 0)
    df['self_to_self'] = np.where(df['Account'] == df['Account.1'], 1, 0)

    # Rename for easier access
    # df.columns = [
    #     'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
    #     'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
    #     'payment_format', 'is_laundering'
    # ]

    df.rename(columns={
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",
        "To Bank": "to_bank",
        "Account.1": "to_account",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering"
    }, inplace=True) 

    if bank_filter is not None: # Filter by from_bank if needed -- vaibhav edit 
        df = df[(df["from_bank"] == bank_filter) | (df["to_bank"] == bank_filter)]

    # Time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['tx_hour'] = df['timestamp'].dt.hour
    df['tx_day'] = df['timestamp'].dt.dayofweek
    df['tx_month'] = df['timestamp'].dt.month

    # Encode categorical fields
    from sklearn.preprocessing import LabelEncoder

    # Encode categorical fields
    for col in ['payment_format', 'receiving_currency', 'payment_currency']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # # One-hot encode categorical fields - Arron
    # categorical_cols = ['payment_format', 'receiving_currency', 'payment_currency'] # Removed from bank and to bank as discussed
    # df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # # Dynamically get the names of the dummy variables - Arron
    # dummy_columns = [col for col in df.columns if any(prefix in col for prefix in categorical_cols)]

    # Feature set
    features = [
        'amount_received', 'amount_paid', 'tx_hour', 'tx_day', 'tx_month',
        'payment_format', 'currency_change', 'self_to_self',
        'receiving_currency', 'payment_currency', # Removed from bank and to bank as discussed
    ] # Can add from_bank, to_bank if needed
    target = 'is_laundering'

    # Drop missing values
    df_model = df[features + [target]].dropna()

    torch.manual_seed(42) # Added random seed - Arron

    # pos_count = (df_model[target] == 1).sum()
    # neg_count = (df_model[target] == 0).sum()
    # pos_weight = min(neg_count / pos_count, 10.0) if pos_count > 0 else 1.0
    positive_weight = (df_model['is_laundering'] == 0).sum() / (df_model['is_laundering'] == 1).sum()
    print('poswt calculated ----> : ', positive_weight)
    # positive_weight = min(positive_weight, 10.0) 
    # print('after scaling down poswt: ', positive_weight)
    pos_weight_tensor = torch.tensor([positive_weight], dtype=torch.float32)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_model[features])
    y = df_model[target].astype(np.float32).values.reshape(-1, 1)

    # # Add small Gaussian noise to features
    # noise = np.random.normal(0, 0.01, X.shape)
    # X += noise

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # dataset = TensorDataset(X_tensor, y_tensor)

    # # Train/Val/Test split
    # train_size = int(0.6 * len(dataset))
    # val_size = int(0.2 * len(dataset))
    # test_size = len(dataset) - train_size - val_size

    # generator = torch.Generator().manual_seed(42)
    # train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # -------- Stratified Split starts here --------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tensor, y_tensor, test_size=0.4, random_state=42, stratify=y_tensor
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl, pos_weight_tensor #vaib -> return positive weight as well