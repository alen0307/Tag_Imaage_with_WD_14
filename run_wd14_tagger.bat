@echo off
chcp 65001 >nul
echo ========================================
echo   WD14 图片标签工具
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
python -c "import onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo [警告] onnxruntime 未安装
    echo.
    set /p install="是否现在安装 onnxruntime? (y/n): "
    if /i "%install%"=="y" (
        echo 正在安装 onnxruntime...
        pip install onnxruntime
        if errorlevel 1 (
            echo [错误] 安装失败
            pause
            exit /b 1
        )
        echo [成功] onnxruntime 安装完成
    ) else (
        echo [提示] 跳过安装，程序可能无法正常运行
    )
) else (
    echo [成功] onnxruntime 已安装
)

echo.
echo [2/3] 测试模型...
python test_wd14.py
if errorlevel 1 (
    echo.
    echo [警告] 模型测试失败，请检查模型文件是否存在
    echo 模型目录: E:\ComfyUI-aki-v1.7\ComfyUI\custom_nodes\comfyui-WD14-Tagger\models
    echo.
    set /p continue="是否继续启动程序? (y/n): "
    if /i not "%continue%"=="y" (
        pause
        exit /b 1
    )
)

echo.
echo [3/3] 启动程序...
echo.
python image_taggerC.py

if errorlevel 1 (
    echo.
    echo [错误] 程序运行出错
    pause
)
