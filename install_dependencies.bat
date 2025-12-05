@echo off
chcp 65001
title 安装依赖
echo ===================================================
echo          正在安装/更新项目依赖...
echo ===================================================
echo.

:: 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 python 命令。请先安装 Python。
    pause
    exit /b
)

:: 升级 pip (可选，但推荐)
echo 正在检查 pip 版本...
python -m pip install --upgrade pip

:: 安装依赖
echo 正在安装依赖库...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [错误] 依赖安装失败。请检查网络连接。
) else (
    echo.
    echo [成功] 依赖安装完成！
    echo 现在你可以运行 run.bat 来启动程序了。
)
pause
