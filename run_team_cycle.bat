@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"
title Team Research Pipeline

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Khong tim thay lenh python trong PATH.
    echo Hay kich hoat dung moi truong Python truoc khi chay file nay.
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo              TEAM RESEARCH PIPELINE MENU
echo ============================================================
echo [1] Crawl Reddit data
echo [2] Prepare Reddit batch for Label Studio
echo [3] Merge Label Studio export into master
echo [4] Enrich labeled data with comment tree
echo [5] Build graphs
echo [6] Train model
echo [0] Exit
echo ============================================================
set /p CHOICE=Chon phase can chay: 

if "%CHOICE%"=="1" goto crawl
if "%CHOICE%"=="2" goto prepare
if "%CHOICE%"=="3" goto merge
if "%CHOICE%"=="4" goto enrich
if "%CHOICE%"=="5" goto graphs
if "%CHOICE%"=="6" goto train
if "%CHOICE%"=="0" goto end

echo Lua chon khong hop le.
pause
goto menu

:crawl
set LIMIT=25
set /p LIMIT=So bai moi moi subreddit [25]: 
if "%LIMIT%"=="" set LIMIT=25

set IMG_ONLY=
set /p IMG_ONLY=Chi lay bai co anh? (y/n) [n]: 

set CMD=python src\utils\research_pipeline.py crawl --limit %LIMIT%
if /I "%IMG_ONLY%"=="y" set CMD=%CMD% --images-only

echo.
echo Running: %CMD%
call %CMD%
pause
goto menu

:prepare
set MODE=
set /p MODE=Chay AUTO hay MANUAL? (auto/manual) [auto]: 
if "%MODE%"=="" set MODE=auto

if /I "%MODE%"=="manual" (
    set START=
    set COUNT=
    set /p START=Start index: 
    set /p COUNT=Count: 
    if "%START%"=="" goto missing_prepare
    if "%COUNT%"=="" goto missing_prepare
    set CMD=python src\utils\research_pipeline.py prepare-label --start %START% --count %COUNT%
) else (
    set CMD=python src\utils\research_pipeline.py prepare-label
)

echo.
echo Running: %CMD%
call %CMD%
pause
goto menu

:missing_prepare
echo Start va Count khong duoc de trong khi chay MANUAL.
pause
goto menu

:merge
set EXPORT_JSON=
set /p EXPORT_JSON=Nhap duong dan file export JSON tu Label Studio: 
if "%EXPORT_JSON%"=="" (
    echo Ban chua nhap duong dan file export.
    pause
    goto menu
)

set CMD=python src\utils\research_pipeline.py merge-labels --input "%EXPORT_JSON%"
echo.
echo Running: %CMD%
call %CMD%
pause
goto menu

:enrich
set LIMIT=
set /p LIMIT=Gioi han so bai moi can enrich (bo trong de chay het): 
set DELAY=2.0
set /p DELAY=Delay giua cac bai [2.0]: 
if "%DELAY%"=="" set DELAY=2.0

set CMD=python src\utils\research_pipeline.py enrich --delay %DELAY%
if not "%LIMIT%"=="" set CMD=%CMD% --limit %LIMIT%

echo.
echo Running: %CMD%
call %CMD%
pause
goto menu

:graphs
set MULTI=
set /p MULTI=Build them multimodal graphs? (y/n) [n]: 

set CMD=python src\utils\research_pipeline.py build-graphs
if /I "%MULTI%"=="y" set CMD=%CMD% --multimodal

echo.
echo Running: %CMD%
call %CMD%
pause
goto menu

:train
echo.
echo Chon model:
echo   [1] baseline_text
echo   [2] baseline_image
echo   [3] baseline_fusion
echo   [4] gnn
echo   [5] multimodal_gnn
set /p MODEL_CHOICE=Lua chon [5]: 
if "%MODEL_CHOICE%"=="" set MODEL_CHOICE=5

set MODEL_NAME=multimodal_gnn
if "%MODEL_CHOICE%"=="1" set MODEL_NAME=baseline_text
if "%MODEL_CHOICE%"=="2" set MODEL_NAME=baseline_image
if "%MODEL_CHOICE%"=="3" set MODEL_NAME=baseline_fusion
if "%MODEL_CHOICE%"=="4" set MODEL_NAME=gnn
if "%MODEL_CHOICE%"=="5" set MODEL_NAME=multimodal_gnn

set EPOCHS=30
set BATCH=32
set /p EPOCHS=Epochs [30]: 
if "%EPOCHS%"=="" set EPOCHS=30
set /p BATCH=Batch size [32]: 
if "%BATCH%"=="" set BATCH=32

set CMD=python src\utils\research_pipeline.py train --model %MODEL_NAME% --epochs %EPOCHS% --batch_size %BATCH%

echo.
echo Running: %CMD%
call %CMD%
pause
goto menu

:end
echo Thoat pipeline menu.
endlocal
exit /b 0
