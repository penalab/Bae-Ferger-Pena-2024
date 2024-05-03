import pathlib
from socket import gethostname

hostname = gethostname()
if hostname == "KENES1A":
    datadir_ot = pathlib.Path(r"E:\FreeField-Andrea")
    datadir_icx = pathlib.Path(r"E:\FreeField-Roland")
elif hostname == "penalab-andrea":
    datadir_ot = pathlib.Path(r"C:\Users\andrea\Documents\Data\FreeField")
    datadir_icx = pathlib.Path(r"C:\Users\andrea\Documents\Data\ICX-FreeField")
else:
    raise ValueError(
        f"Please set up the approriate path for your computer. Your host name is {hostname}"
    )

units_ot = pathlib.Path("./curated_data/units_ot.csv")
units_icx = pathlib.Path("./curated_data/units_icx.csv")
srf_dates_ot = pathlib.Path("./curated_data/srf_dates_ot.csv")
srf_dates_icx = pathlib.Path("./curated_data/srf_dates_icx.csv")
