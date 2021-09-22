from setuptools import setup, find_packages

setup(name='maddpg',
      version='0.0.1',
      description='Multi-Agent Deep Deterministic Policy Gradient',
      url='https://github.com/openai/maddpg',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(), #给packages指定find_packages()，意为包括在本目录下的所有Python包。
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
'''
setup函数包含的参数解释
--name 包名称------------生成的egg名称
--version (-V) 包版本----生成egg包的版本号
--author 程序的作者------包的制作者名字
--author_email 程序的作者的邮箱地址
--maintainer 维护者
--maintainer_email 维护者的邮箱地址
--url 程序的官网地址
--license 程序的授权信息
--description 程序的简单描述-------程序的概要介绍
--long_description 程序的详细描述---程序的详细描述
--platforms 程序适用的软件平台列表
--classifiers 程序的所属分类列表
--keywords 程序的关键字列表
--packages 需要处理的包目录（包含__init__.py的文件夹）-------和setup.py同一目录下搜索各个含有 init.py的包
--py_modules 需要打包的python文件列表
--download_url 程序的下载地址
--cmdclass
--data_files 打包时需要打包的数据文件，如图片，配置文件等
--scripts 安装时需要执行的脚步列表
--package_dir 告诉setuptools哪些目录下的文件被映射到哪个源码包。一个例子：package_dir = {'': 'lib'}，表示“root package”中的模块都在lib 目录中。
--requires 定义依赖哪些模块
--provides定义可以为哪些模块提供依赖
--find_packages() 对于简单工程来说，手动增加packages参数很容易，刚刚我们用到了这个函数，它默认在和setup.py同一目录下搜索各个含有 init.py的包。
其实我们可以将包统一放在一个src目录中，另外，这个包内可能还有aaa.txt文件和data数据文件夹。另外，也可以排除一些特定的包
find_packages(exclude=[".tests", ".tests.", "tests.", "tests"])
--install_requires = ["requests"] 需要安装的依赖包
--entry_points 动态发现服务和插件
'''
