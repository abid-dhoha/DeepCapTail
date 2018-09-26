from setuptools import setup

setup(
    name='DeepCapTail',
    version='0.0.1',
    package_dir={
        'code': 'DeepCapTail/code'
    },
    packages=['code'],
    install_requires=['biopython==1.7', 'scikit-learn==0.19.1', 'pandas==0.22.0'],
    entrypoints={
        'console_scripts':
            [
                'train_save_models=code.train_save_model:main'
            ]
    }
)
