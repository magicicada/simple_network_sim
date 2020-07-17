from io import TextIOWrapper

import pandas as pd
from data_pipeline_api.file_api import FileAPI


class Datastore(FileAPI):
    def read_table(self, data_product: str) -> pd.DataFrame:
        with TextIOWrapper(self.open_for_read(data_product=data_product, extension="csv")) as csv_file:
            return pd.read_csv(csv_file)

    def write_table(self, data_product: str, value: pd.DataFrame):
        with TextIOWrapper(self.open_for_write(data_product=data_product, extension="csv")) as csv_file:
            value.to_csv(csv_file, index=False)
