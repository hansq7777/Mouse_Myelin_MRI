@echo off
REM Launch the MT pipeline UI with a visible console so errors don't vanish.
setlocal
pushd "%~dp0"

set LAUNCHER=
for %%P in (python.exe py.exe) do (
  where %%P >nul 2>nul
  if not errorlevel 1 if not defined LAUNCHER set LAUNCHER=%%P
)

if not defined LAUNCHER (
  echo Could not find python or py on PATH. Please install Python 3 and add it to PATH.
  pause
  exit /b 1
)

if /i "%LAUNCHER%"=="py.exe" (
  %LAUNCHER% -3 -m ui_app
) else (
  %LAUNCHER% -m ui_app
)

if errorlevel 1 (
  echo UI failed to start. Check messages above.
  pause
  goto :eof
)

if exist "ui_app\close_console.flag" (
  del /f /q "ui_app\close_console.flag" >nul 2>nul
  goto :eof
)

pause
popd
endlocal
