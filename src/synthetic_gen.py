import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

def build_fake_data(data, n):
    # get schema
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(data)
    
    # train copula
    copula = GaussianCopulaSynthesizer(meta)
    copula.fit(data)
    
    # generate rows
    fake = copula.sample(num_rows=n)
    
    return fake, meta