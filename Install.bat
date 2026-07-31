@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "INSTALL_LOG=%CD%\install.log"
> "%INSTALL_LOG%" echo Mistake Notebook installation started at %DATE% %TIME%

echo.
echo ============================================================
echo Mistake Notebook - Windows installer
echo ============================================================
echo Log: %INSTALL_LOG%
echo The first installation downloads about 6-10GB.
echo Keep this window open and maintain a stable network connection.
echo.

call :ensure_winget
if errorlevel 1 goto :failed
call :ensure_git
if errorlevel 1 goto :failed
call :ensure_python
if errorlevel 1 goto :failed

echo [4/7] Creating the Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
    "%PYTHON_EXE%" -m venv ".venv" >> "%INSTALL_LOG%" 2>&1
    if errorlevel 1 goto :failed
)
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

echo [5/7] Installing project dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VENV_PYTHON%" -m pip install -e ".[dev,v2]"
if errorlevel 1 goto :failed

echo [6/7] Installing OCR, formula, and document models...
echo This step can take a long time. Re-run this installer after an interruption.
"%VENV_PYTHON%" scripts\setup_runtime.py
if errorlevel 1 goto :failed

echo [7/7] Running installation checks...
"%VENV_PYTHON%" -c "from mistake_book.app import create_app; app=create_app(); print('Application import: OK')"
if errorlevel 1 goto :failed
"%VENV_PYTHON%" -c "from mistake_book.font_selection import default_font_metrics; print('Chinese font:', default_font_metrics()['rendered_family'])"
if errorlevel 1 goto :failed
"%VENV_PYTHON%" -m pytest -q tests\test_portability.py
if errorlevel 1 goto :failed

>> "%INSTALL_LOG%" echo Installation completed at %DATE% %TIME%
echo.
echo ============================================================
echo Installation complete
echo ============================================================
echo Start the application with:
echo   .venv\Scripts\mistake-book.exe --root .
echo.
echo Then open http://127.0.0.1:8765
echo The installer is safe to run again.
echo.
pause
exit /b 0

:ensure_winget
echo [1/7] Checking Windows Package Manager...
where winget >nul 2>&1
if errorlevel 1 (
    echo winget was not found.
    echo Install "App Installer" from Microsoft Store, then run this file again.
    exit /b 1
)
exit /b 0

:ensure_git
echo [2/7] Checking Git...
where git >nul 2>&1
if errorlevel 1 (
    echo Installing Git with winget...
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 exit /b 1
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
)
git --version
exit /b %ERRORLEVEL%

:ensure_python
echo [3/7] Checking Python 3.11...
py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
    exit /b 0
)

echo Installing Python 3.11 with winget...
winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 1

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
) else (
    py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
    if errorlevel 1 (
        echo Python was installed but is not visible in this terminal.
        echo Close this window and run Install.bat again.
        exit /b 1
    )
    for /f "delims=" %%I in ('py -3.11 -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%I"
)
exit /b 0

:failed
>> "%INSTALL_LOG%" echo Installation failed at %DATE% %TIME%
echo.
echo ============================================================
echo Installation failed
echo ============================================================
echo Review the error above and the log at:
echo   %INSTALL_LOG%
echo Fix the reported issue, then run Install.bat again.
echo.
pause
exit /b 1
