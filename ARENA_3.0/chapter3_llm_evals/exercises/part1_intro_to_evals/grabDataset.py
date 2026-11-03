import io
import logging
import httpx
import pandas as pd
from tabulate import tabulate
from typing import Optional
from datasets import load_dataset
pd.set_option('display.max_columns', None)

logger = logging.getLogger(__name__)

# Supported file types and their pandas readers
_READERS = {
    ".csv":     pd.read_csv,
    ".tsv":     lambda f, **kw: pd.read_csv(f, sep="\t", **kw),
    ".json":    pd.read_json,
    ".jsonl":   lambda f, **kw: pd.read_json(f, lines=True, **kw),
    ".parquet": pd.read_parquet,
    ".xlsx":    pd.read_excel,
    ".xls":     pd.read_excel,
}

_VALID_DISPLAY_MODES = {None, "tabulate"}

class GrabDataset:
    """
    Generic loader for benchmark datasets from GitHub or Hugging Face.

    Args:
        url:        Raw URL (GitHub/direct) or HF dataset name ("owner/dataset").
        n:          Number of sample rows to display.
        split:      HF split to load (default: "train"). Ignored for GitHub URLs.
        subset:     HF subset/config name (e.g., "en"). Ignored for GitHub URLs.
        encoding:   Character encoding override (default: "utf-8").
        hf_token:   Optional HF access token for gated/private datasets.
        timeout:    HTTP request timeout in seconds (default: 30).
    """

    _GH_BLOB = "github.com"
    _GH_RAW  = "raw.githubusercontent.com"
    _HF_BASE = "huggingface.co/datasets"

    def __init__(
        self,
        url: str,
        n: int = 5,
        split: str = "train",
        subset: Optional[str] = None,
        encoding: str = "utf-8",
        hf_token: Optional[str] = None,
        timeout: int = 30
    ):
        self.url = url.strip()
        self.n = n
        self.split = split
        self.subset = subset
        self.encoding = encoding
        self.hf_token = hf_token
        self.timeout = timeout
        self._df: Optional[pd.DataFrame] = None  # cached after first load


    def load(self) -> "GrabDataset":
        """Fetch and cache the dataset. Returns self for method chaining."""
        
        if self._is_hf_url() or self._is_hf_name():
            self._df = self._load_hf()
        else:
            self._df = self._load_url()
        
        print(f"Loaded {len(self._df)} rows × {len(self._df.columns)} cols")
        
        return self

    def show(self, n: Optional[int] = None, display_mode: Optional[str] = None, seed: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Display or return n sample rows. Auto-loads if not yet loaded.

        Args:
            n: Number of rows to sample. Defaults to self.n.
            display_mode: 'tabulate' prints a table. None returns the DataFrame.
            seed: Random seed for reproducible sampling.
        """
        if display_mode not in _VALID_DISPLAY_MODES:
            raise ValueError(f"Invalid display_mode '{display_mode}'. Choose from: {_VALID_DISPLAY_MODES}")

        if self._df is None:
            self.load()

        sample_n = min(n or self.n, len(self._df))
        sample = self._df.sample(sample_n, random_state=seed) if sample_n > 0 else self._df.head(0)

        print(f"Displaying {sample_n} samples")

        if display_mode == "tabulate":
            print(tabulate(sample.values, headers=self._df.columns.tolist(), maxcolwidths=60))
            return None

        return sample

    @property
    def df(self) -> pd.DataFrame:
        """Access the underlying DataFrame (auto-loads on first access)."""
        
        if self._df is None:
            self.load()
        return self._df

    def _is_hf_url(self) -> bool:
        """Detects huggingface.co/datasets/... URLs."""
        
        return self._HF_BASE in self.url

    def _is_hf_name(self) -> bool:
        """
        Detects bare HF dataset names like 'owner/dataset'
        that are NOT http URLs and NOT local file paths.
        """
        return (
            not self.url.startswith("http")
            and not self.url.startswith("/")
            and "/" in self.url
        )

    def _is_github_blob(self) -> bool:
        """Detects regular GitHub 'blob' URLs that need conversion to raw."""
        
        return self._GH_BLOB in self.url and "/blob/" in self.url

    
    def _load_url(self) -> pd.DataFrame:
        
        raw_url = self._normalise_url(self.url)
        ext = self._infer_extension(raw_url)
        reader = _READERS.get(ext)

        if reader is None:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported: {list(_READERS)}")

        content = self._fetch(raw_url)

        # parquet and xlsx are binary — wrap in BytesIO
        if ext in (".parquet", ".xlsx", ".xls"):
            return reader(io.BytesIO(content))

        text = content.decode(self.encoding, errors="replace")
        return reader(io.StringIO(text))

    def _load_hf(self) -> pd.DataFrame:
        """Load a HF dataset via the `datasets` library."""
        
        # Extract repo id from a full HF URL if needed
        name = self.url
        if self._is_hf_url():
            # e.g. https://huggingface.co/datasets/owner/repo
            parts = self.url.rstrip("/").split("/datasets/", 1)
            name  = parts[1] if len(parts) == 2 else self.url

        try:
            ds = load_dataset(
                name,
                name=self.subset,       # config/subset
                split=self.split,
                token=self.hf_token,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HF dataset '{name}' "
                f"(split='{self.split}', subset='{self.subset}'): {exc}"
            ) from exc

        return ds.to_pandas()


    def _normalise_url(self, url: str) -> str:
        """
        Convert GitHub blob URLs to raw.githubusercontent.com.
        Also appends ?raw=true if the domain is github.com but not a blob.
        """
        if self._is_github_blob():
            # https://github.com/user/repo/blob/branch/file.csv
            # https://raw.githubusercontent.com/user/repo/branch/file.csv
            url = (url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/"))
        elif self._GH_BLOB in url and "?" not in url:
            url = url + "?raw=true"
        return url

    def _infer_extension(self, url: str) -> str:
        """Infer file extension from URL path (strips query params)."""
        
        path = url.split("?")[0].split("#")[0]
        for ext in _READERS:
            if path.lower().endswith(ext):
                return ext
        raise ValueError(f"Cannot infer file type from URL: '{url}'. Rename the file or extend _READERS.")

    def _fetch(self, url: str) -> bytes:
        """
        Fetch raw bytes with:
        - redirect following
        - timeout guard
        - HTTP error surfacing
        - retry on transient failures (429, 5xx)
        """
        transport = httpx.HTTPTransport(retries=3)
        with httpx.Client(
            follow_redirects=True,
            timeout=self.timeout,
            transport=transport,
        ) as client:
            try:
                response = client.get(url)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"HTTP {exc.response.status_code} fetching '{url}'"
                ) from exc
            except httpx.TimeoutException as exc:
                raise TimeoutError(
                    f"Request timed out after {self.timeout}s: '{url}'"
                ) from exc
            except httpx.RequestError as exc:
                raise ConnectionError(
                    f"Network error fetching '{url}': {exc}"
                ) from exc