@echo off
:: 切换到脚本所在目录，防止因路径问题导致找不到文件
cd /d "%~dp0"

:: 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ===================================================
    echo [错误] 未找到 Python！
    echo 请先安装 Python 3.8+ 并勾选 "Add Python to PATH"
    echo ===================================================
    pause
    exit /b
)

:: 运行 Python 启动脚本
python launcher.py

:: 如果 Python 脚本本身崩溃了（极少情况），这里会暂停
if %errorlevel% neq 0 (
    echo.
    echo [系统提示] Python 脚本执行异常。
    pause
)
