@echo off
echo ============================================================
echo  Phishing Detection Pipeline - Full Execution
echo ============================================================

echo.
echo [1/8] Converting PhishTank XML...
python src/convert_phishtank.py
if %errorlevel% neq 0 goto error

echo.
echo [2/8] Converting URLhaus TXT...
python src/convert_urlhaus.py
if %errorlevel% neq 0 goto error

echo.
echo [3/8] Building balanced datasets...
python src/build_dataset.py
if %errorlevel% neq 0 goto error

echo.
echo [4/8] Preparing and cleaning data...
python src/prepare_data.py
if %errorlevel% neq 0 goto error

echo.
echo [5/8] Extracting URL features...
python src/extract_features.py
if %errorlevel% neq 0 goto error

echo.
echo [6/8] Training classification models...
python src/train_models.py
if %errorlevel% neq 0 goto error

echo.
echo [7/8] Running cross-dataset evaluation...
python src/cross_dataset_eval.py
if %errorlevel% neq 0 goto error

echo.
echo [8/8] Running inference validation...
python src/predict.py --url https://www.google.com
if %errorlevel% neq 0 goto error

echo.
echo ============================================================
echo  Pipeline completed successfully.
echo  All outputs saved to outputs/
echo ============================================================
goto end

:error
echo.
echo ============================================================
echo  ERROR: Pipeline failed at the step above.
echo  Check the error message and resolve before continuing.
echo ============================================================
exit /b 1

:end
```
