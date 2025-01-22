# Lambda Python 베이스 이미지 사용
FROM amazon/aws-lambda-python:3.12

# Chrome 및 필수 의존성 설치
RUN dnf install -y atk cups-libs gtk3 libXcomposite alsa-lib \
    libXcursor libXdamage libXext libXi libXrandr libXScrnSaver \
    libXtst pango at-spi2-atk libXt xorg-x11-server-Xvfb \
    xorg-x11-xauth dbus-glib dbus-glib-devel nss mesa-libgbm jq unzip

# Chrome 설치 스크립트 복사 및 실행
COPY ./chrome-installer.sh ./chrome-installer.sh
RUN chmod +x ./chrome-installer.sh && ./chrome-installer.sh && rm ./chrome-installer.sh

# Selenium 설치
RUN pip install selenium

# 애플리케이션 코드 복사
COPY main.py ./

# Lambda 핸들러 실행
CMD ["main.lambda_handler"]
