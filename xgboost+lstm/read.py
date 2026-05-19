import xarray as xr

# 替换成你的nc文件路径
ds = xr.open_dataset("jiangyu.nc")

# 1.打印所有变量、经纬度、时间（看有没有气压/降雨/辐射/比湿）
print(ds)

# 2.单独看某个变量（比如比湿q、降水pre）
print(ds['q'])  