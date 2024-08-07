.PHONY: all

all: linux_requirements python_requirements MFA_pretrained MFA_pretrained_chinese
# all: MFA_thirdparty MFA_pretrained_chinese

linux_requirements:
	apt-get update && apt-get install -y libsndfile1 libopenblas-dev

python_requirements:
	conda install -c conda-forge -y --file conda_requirements.txt
	pip install pip setuptools --upgrade
	pip install -e ../.[pytorch] \
		--find-links https://download.pytorch.org/whl/torch_stable.html

MFA_thirdparty:
	mfa thirdparty download
	mfa thirdparty validate

MFA_pretrained:
	mfa model download acoustic english_mfa
	mfa model download g2p english_us_mfa
	mfa model download dictionary english_us_mfa

MFA_pretrained_chinese:
	mfa model download acoustic mandarin_mfa
	mfa model download g2p mandarin_china_pinyin_mfa
	mfa model download dictionary mandarin_pinyin

