from utils.config import logger
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, RequestException
import pandas as pd
from utils.s3_utils import ingest_df_raw_zone_s3

def define_path(path: Path = None) -> Path:
    """
    Define and create an output directory.

    If no path is provided, a "data" directory in the current working directory
    is created. Otherwise, the provided path is resolved to an absolute path.

    Parameters
    ----------
    path : pathlib.Path, optional
        The target directory path. If None, defaults to Path.cwd() / "data".

    Returns
    -------
    pathlib.Path
        The resolved and created output directory.

    Raises
    ------
    PermissionError
        If the directory cannot be created due to permission issues.
    OSError
        If the directory cannot be created for any other OS-related reason.
    """
    if path is None:
        path = Path.cwd() / "data"
    else:
        path = Path(path).resolve()
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {path}")
    except PermissionError as e:
        logger.error(f"Permission denied creating directory {path}: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise
    
    return path


def authenticate_kaggle_api() -> KaggleApi:
    """
    Authenticate and return a Kaggle API client.

    This function initializes the KaggleApi object and attempts to authenticate
    using the user's stored Kaggle credentials.

    Returns
    -------
    KaggleApi
        An authenticated Kaggle API client.

    Raises
    ------
    RuntimeError
        If authentication fails due to invalid credentials or missing configuration.
    """
    api = KaggleApi()
    try:
        api.authenticate()
        logger.debug("Authenticated with Kaggle AP.")
    except RequestException as e:
        logger.error(f"Kaggle API authentication failed: {e}")
        raise RuntimeError("Kaggle authentication failed") from e
    
    return api


def get_dataset_path(
    dataset: str = "priyamchoksi/credit-card-transactions-dataset",
    data_path: Path = None
) -> Path:
    """
    Download and extract a Kaggle dataset if not already present.

    Checks if the target directory contains any files. If empty, downloads
    the specified dataset from Kaggle, extracts it, and returns the path.

    Parameters
    ----------
    dataset : str, optional
        The Kaggle dataset identifier in the format "username/dataset-name".
        Defaults to "priyamchoksi/credit-card-transactions-dataset".
    data_path : pathlib.Path, optional
        Directory to which the dataset should be downloaded. If None,
        defaults to the "data" directory in the current working directory.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the dataset files.

    Raises
    ------
    RuntimeError
        If the download or extraction fails.
    """
    data_path = define_path(path=data_path)
    api = authenticate_kaggle_api()
    
    if any(data_path.iterdir()):
        logger.info(f"Dataset already exists in {data_path}")
        return data_path
    
    try:
        logger.info(f"Downloading dataset '{dataset}' to {data_path}")
        api.dataset_download_files(
            dataset,
            path=str(data_path),
            unzip=True,
            quiet=False
        )
        logger.info("Dataset downloaded successfully")
    except Exception as e:
        logger.error(f"Kaggle API download failed: {e}")
        raise RuntimeError(f"Failed to download dataset {dataset}") from e

    return data_path


def list_dataset_files(data_path: Path) -> list:
    """
    List all files in a given directory.

    Parameters
    ----------
    data_path : pathlib.Path
        Path to the directory whose files will be listed.

    Returns
    -------
    list of pathlib.Path
        A list of file paths contained in the directory.
    """
    return [f for f in data_path.iterdir() if f.is_file()]

def loading_raw_data(
    csv_path: Path,
    ingest_fmt: str = "csv",
    **read_csv_kwargs
) -> pd.DataFrame:
    """
    Load raw credit card transaction data from a CSV file and ingest to S3.

    Parameters
    ----------
    csv_path : pathlib.Path or str
        Directory path where the `credit_card_transactions.{ingest_fmt}` file 
        is located. If a directory is provided, the filename will be automatically
        appended in the format `credit_card_transactions.{ingest_fmt}`.
    ingest_fmt : str, optional
        File format extension for the input data (default: 'csv'). Used to
        construct the full filename as `credit_card_transactions.{ingest_fmt}`.
    **read_csv_kwargs : dict, optional
        Additional keyword arguments to pass to `pd.read_csv()`. Overrides default
        read parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded transaction data.

    Raises
    ------
    FileNotFoundError
        If the constructed file path does not exist.
    pd.errors.ParserError
        If there are issues parsing the CSV file.
    """
    csv_path = Path(csv_path / f"credit_card_transactions.{ingest_fmt}")

    logger.info(f"Loading raw data from {csv_path}.")
    try:
        defaults = dict(on_bad_lines="warn")
        params = {**defaults, **read_csv_kwargs}
        df = pd.read_csv(csv_path, **params)
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parsing failed for {csv_path}: {e}")
        raise

    logger.info(f"Successfully loaded {len(df):,} rows.")
    ingest_df_raw_zone_s3(
        df=df,
        source="kaggle",
        fmt=ingest_fmt
    )
    logger.info("Raw data ingestion to S3 completed.")
    return df