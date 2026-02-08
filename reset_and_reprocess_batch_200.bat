@echo off
echo ========================================================
echo       NUCLEAR RESET: BATCH 200-400 REPROCESSING
echo ========================================================

echo [1/8] Stopping Label Studio Docker...
docker-compose down
echo.

echo [2/8] Cleaning up old processed data...
rem Xoa file processed cu (giu lai folder)
if exist "data\02_processed\dataset_output.jsonl" del /F /Q "data\02_processed\dataset_output.jsonl"
if exist "data\02_processed\images\Fakeddit_*" rmdir /S /Q "data\02_processed\images"
mkdir "data\02_processed\images"

rem Xoa clean data cua batch 200_400
if exist "data\03_clean\Fakeddit\batch_200_400" rmdir /S /Q "data\03_clean\Fakeddit\batch_200_400"
echo.

echo [3/8] Extracting Batch 200-400 from Raw...
python src/utils/batch_extractor.py --start 200 --count 200
echo.

echo [4/8] Processing Images (Download & Resize)...
python src/data/fakeddit_preprocessor_image.py --input "data/01_raw/Fakeddit/batches/Fakeddit_200_400.jsonl"
echo.

echo [5/8] Processing Text & Generating Metadata...
python src/data/fakeddit_process_text.py --batch-name batch_200_400
echo.

echo [6/8] Fixing Paths for Label Studio Docker...
python src/utils/convert_to_ls_json.py --batch-name batch_200_400 --docker
echo.

echo [7/8] Restarting Docker (Clean State)...
rem Xoa luon volume cu ne
rd /s /q labels_storage\export
del /f /q labels_storage\label_studio.sqlite3

docker-compose up -d
echo.

echo ========================================================
echo [8/8] DONE! PLEASE FOLLOW THESE STEPS:
echo 1. Wait 10-15 seconds for Label Studio to start.
echo 2. Go to   
echo 3. Create a NEW Account & Project.
echo 4. Import file: data\03_clean\Fakeddit\batch_200_400\train_for_ls.json
echo ========================================================
pause
