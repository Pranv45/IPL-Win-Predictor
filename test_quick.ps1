# Quick Test Script for IPL Win Predictor
# Run this script to test each component step by step

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IPL Win Predictor - Quick Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion" -ForegroundColor Green

if (Test-Path "data\raw\matches.csv") {
    Write-Host "✓ Data files found" -ForegroundColor Green
} else {
    Write-Host "✗ Data files missing!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Menu:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Test Data Processing (Step 1)"
Write-Host "2. Test Feature Engineering (Step 2)"
Write-Host "3. Test Model Training (Step 3)"
Write-Host "4. Test Model Evaluation (Step 4)"
Write-Host "5. Test API - Health Check (Step 5a)"
Write-Host "6. Test API - Prediction (Step 5b) - CRITICAL LABEL ENCODER TEST"
Write-Host "7. Test API - Metrics (Step 5c)"
Write-Host "8. Run All Tests in Sequence"
Write-Host "9. Check Generated Files"
Write-Host "0. Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (0-9)"

switch ($choice) {
    "1" {
        Write-Host "`nRunning Data Processing..." -ForegroundColor Yellow
        python scripts/ingest_and_process.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Data Processing completed!" -ForegroundColor Green
            if (Test-Path "data\processed\processed_ipl_data.parquet") {
                Write-Host "✓ Output file created" -ForegroundColor Green
            }
        } else {
            Write-Host "`n✗ Data Processing failed!" -ForegroundColor Red
        }
    }
    "2" {
        Write-Host "`nRunning Feature Engineering..." -ForegroundColor Yellow
        python scripts/feature_engineering.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Feature Engineering completed!" -ForegroundColor Green
            if (Test-Path "data\features\engineered_features.parquet") {
                Write-Host "✓ Features file created" -ForegroundColor Green
            }
            if (Test-Path "data\features\label_encoders.pkl") {
                Write-Host "✓ Label encoders created" -ForegroundColor Green
            }
        } else {
            Write-Host "`n✗ Feature Engineering failed!" -ForegroundColor Red
        }
    }
    "3" {
        Write-Host "`nRunning Model Training..." -ForegroundColor Yellow
        python scripts/train_model.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Model Training completed!" -ForegroundColor Green
            if (Test-Path "models\ipl_win_predictor.pkl") {
                Write-Host "✓ Model file created" -ForegroundColor Green
            }
        } else {
            Write-Host "`n✗ Model Training failed!" -ForegroundColor Red
        }
    }
    "4" {
        Write-Host "`nRunning Model Evaluation..." -ForegroundColor Yellow
        python scripts/evaluate_model.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Model Evaluation completed!" -ForegroundColor Green
            if (Test-Path "metrics\evaluation_results.json") {
                Write-Host "✓ Evaluation results created" -ForegroundColor Green
            }
        } else {
            Write-Host "`n✗ Model Evaluation failed!" -ForegroundColor Red
        }
    }
    "5" {
        Write-Host "`nTesting API Health Check..." -ForegroundColor Yellow
        Write-Host "Make sure API is running first!" -ForegroundColor Yellow
        Write-Host "Start API with: python -m uvicorn api.main:app --host 0.0.0.0 --port 8080" -ForegroundColor Yellow
        Write-Host ""
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ API Health Check passed!" -ForegroundColor Green
            Write-Host "Response: $($response.Content)" -ForegroundColor Green
        } else {
            Write-Host "✗ API Health Check failed!" -ForegroundColor Red
            Write-Host "Make sure API is running on port 8080" -ForegroundColor Yellow
        }
    }
    "6" {
        Write-Host "`nTesting API Prediction (LABEL ENCODER FIX TEST)..." -ForegroundColor Yellow
        Write-Host "Make sure API is running first!" -ForegroundColor Yellow
        Write-Host ""
        $body = @{
            team1 = "Mumbai Indians"
            team2 = "Chennai Super Kings"
            venue = "Wankhede Stadium"
            city = "Mumbai"
            team1_win_percentage = 0.65
            team2_win_percentage = 0.55
            team1_recent_form = 0.7
            team2_recent_form = 0.6
            team1_head_to_head = 0.6
            team2_head_to_head = 0.4
        } | ConvertTo-Json

        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8080/predict" -Method POST -Body $body -ContentType "application/json"
            Write-Host "✓ Prediction successful!" -ForegroundColor Green
            Write-Host "Response:" -ForegroundColor Green
            $response | ConvertTo-Json -Depth 10
            Write-Host "`n✓ LABEL ENCODER FIX WORKS!" -ForegroundColor Green
        } catch {
            Write-Host "✗ Prediction failed!" -ForegroundColor Red
            Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
            if ($_.Exception.Message -like "*transform*" -or $_.Exception.Message -like "*label*") {
                Write-Host "`n⚠ LABEL ENCODER BUG STILL EXISTS!" -ForegroundColor Red
                Write-Host "Check api/main.py - label encoders should use .get() not .transform()" -ForegroundColor Yellow
            }
        }
    }
    "7" {
        Write-Host "`nTesting API Metrics Endpoint..." -ForegroundColor Yellow
        $response = Invoke-WebRequest -Uri "http://localhost:8080/metrics" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Metrics endpoint works!" -ForegroundColor Green
            Write-Host "Metrics type: $($response.Headers['Content-Type'])" -ForegroundColor Green
        } else {
            Write-Host "✗ Metrics endpoint failed!" -ForegroundColor Red
        }
    }
    "8" {
        Write-Host "`nRunning All Tests in Sequence..." -ForegroundColor Cyan
        Write-Host "This will take a few minutes..." -ForegroundColor Yellow
        Write-Host ""

        # Step 1
        Write-Host "[1/4] Data Processing..." -ForegroundColor Yellow
        python scripts/ingest_and_process.py
        if ($LASTEXITCODE -ne 0) { Write-Host "✗ Failed at Step 1" -ForegroundColor Red; exit 1 }

        # Step 2
        Write-Host "[2/4] Feature Engineering..." -ForegroundColor Yellow
        python scripts/feature_engineering.py
        if ($LASTEXITCODE -ne 0) { Write-Host "✗ Failed at Step 2" -ForegroundColor Red; exit 1 }

        # Step 3
        Write-Host "[3/4] Model Training..." -ForegroundColor Yellow
        python scripts/train_model.py
        if ($LASTEXITCODE -ne 0) { Write-Host "✗ Failed at Step 3" -ForegroundColor Red; exit 1 }

        # Step 4
        Write-Host "[4/4] Model Evaluation..." -ForegroundColor Yellow
        python scripts/evaluate_model.py
        if ($LASTEXITCODE -ne 0) { Write-Host "✗ Failed at Step 4" -ForegroundColor Red; exit 1 }

        Write-Host "`n✓ All pipeline steps completed!" -ForegroundColor Green
    }
    "9" {
        Write-Host "`nChecking Generated Files..." -ForegroundColor Yellow
        Write-Host ""

        $files = @(
            "data\processed\processed_ipl_data.parquet",
            "data\features\engineered_features.parquet",
            "data\features\label_encoders.pkl",
            "models\ipl_win_predictor.pkl",
            "models\scaler.pkl",
            "models\model_info.json",
            "metrics\evaluation_results.json"
        )

        foreach ($file in $files) {
            if (Test-Path $file) {
                $size = (Get-Item $file).Length
                Write-Host "✓ $file ($([math]::Round($size/1KB, 2)) KB)" -ForegroundColor Green
            } else {
                Write-Host "✗ $file (MISSING)" -ForegroundColor Red
            }
        }
    }
    "0" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid choice!" -ForegroundColor Red
    }
}

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

