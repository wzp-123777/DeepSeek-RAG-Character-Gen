import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """检查包是否已安装"""
    if importlib.util.find_spec(package_name) is None:
        return False
    return True

def install_dependencies():
    """安装依赖"""
    print("发现缺少依赖，正在自动安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖安装完成！")
    except subprocess.CalledProcessError:
        print("错误：依赖安装失败。请检查网络连接或手动运行 install_dependencies.bat")
        raise

def main():
    os.system("title DeepSeek RAG 启动器")
    print("===================================================")
    print("          DeepSeek RAG 角色生成器启动程序")
    print("===================================================")
    print(f"Python 路径: {sys.executable}")
    print("正在检查环境配置...\n")
    
    if not os.path.exists("requirements.txt"):
        print("错误：当前目录下找不到 requirements.txt 文件。")
        input("按回车键退出...")
        return

    # 检查关键依赖，如果没有安装则自动安装
    required_packages = ["streamlit", "openai", "langchain_community", "chromadb"]
    missing_packages = []
    for pkg in required_packages:
        # 处理一些包名和导入名不一致的情况
        import_name = pkg
        if pkg == "langchain_community": import_name = "langchain_community"
        
        if not check_package(import_name):
            missing_packages.append(pkg)

    if missing_packages:
        print(f"检测到缺少以下依赖: {', '.join(missing_packages)}")
        try:
            install_dependencies()
        except:
            input("按回车键退出...")
            return

    print("\n环境检查通过。正在启动 Streamlit 应用...")
    print("请稍候，浏览器将自动打开...")
    print("如果浏览器没有反应，请手动访问: http://localhost:8501")
    print("===================================================\n")

    # 启动 Streamlit
    # 使用 sys.executable 确保使用当前的 Python 解释器
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    
    try:
        # 使用 subprocess.run 运行，这样可以捕获退出码
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[警告] 应用异常退出，退出码: {result.returncode}")
    except KeyboardInterrupt:
        print("\n用户手动停止了应用。")
    except Exception as e:
        print(f"\n[错误] 启动过程中发生异常: {e}")
    
    print("\n===================================================")
    print("程序运行结束。")
    input("按回车键关闭窗口...")

if __name__ == "__main__":
    main()
