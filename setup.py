from setuptools import setup

setup(
      name="statDIST",
      version="1.0",
      description="Statistical distances for samples and random variables",
      author="Ziad Ghauch",
      url="https://github.com/ZGGhauch/statDIST",
      packages=["statDIST"],
      include_package_data=True,
      requires=["numpy","scipy","sklearn","statsmodels"]
)
