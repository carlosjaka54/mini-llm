@echo off
rem Inicia la interfaz web del mini-LLM con doble clic (activa el entorno virtual solo).
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo No se encontro el entorno virtual .venv en esta carpeta.
  echo Crealo con: python -m venv .venv
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
mini-llm-web %*
pause
