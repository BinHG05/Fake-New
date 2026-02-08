@echo off
echo ========================================================
echo       MERGE LABELED DATA: BATCH 200-400
echo ========================================================
echo.
echo Please make sure you have exported your data from Label Studio!
echo Expected file name: project-*-export_*.json 
echo (You should rename it to 'export_batch_200.json' and place it in 'labels_storage/export' for this script to find it, OR drag and drop it here)
echo.

set /p INPUT_FILE="Drag and drop your export JSON file here and press Enter (or type path): "

if not exist %INPUT_FILE% (
    echo ❌ Error: File not found!
    pause
    exit /b
)

echo.
echo [1/2] Converting Export JSON to JSONL and Appending to Master...
python src/utils/convert_ls_export_to_jsonl.py "%INPUT_FILE%" "data/03_clean/Fakeddit/labeled_master.jsonl" --append

echo.
echo ========================================================
echo [2/2] DONE! Data merged into 'data/03_clean/Fakeddit/labeled_master.jsonl'
echo ========================================================
pause
