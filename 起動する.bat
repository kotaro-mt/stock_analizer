@echo off
chcp 65001 > nul
echo ========================================================
echo   株価トレンドライン自動検出アプリを起動しています...
echo ========================================================
cd /d "c:\Users\matsu\OneDrive\claude\stock_future"
"C:\Users\matsu\anaconda3\python.exe" -m streamlit run chart_app.py
pause
