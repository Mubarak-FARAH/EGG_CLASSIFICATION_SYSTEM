# Eggcellent ID - team setup package

This package is for Windows teammates.

## What is inside

- `app.py`
- `requirements.txt`
- `install_app.bat`
- `run_app.bat`
- `start_ngrok.bat`
- project folders like `APP_ARCHITECTURE`, `MODEL`, and `Branding`

## First-time setup

1. Install Python 3.11 or newer.
2. Put this package folder anywhere on the computer.
3. Double-click `install_app.bat`.

This will:
- create a local virtual environment named `.venv`
- install all Python dependencies from `requirements.txt`

## Running the app

Double-click `run_app.bat`.

The app will open locally in the browser on port 8501.

## Testing on phone

To let a phone open the app with a secure HTTPS URL:

1. Install ngrok.
2. Authenticate ngrok one time with:
   `ngrok config add-authtoken YOUR_TOKEN`
3. Start the app with `run_app.bat`
4. In another terminal or by double-clicking, run `start_ngrok.bat`
5. Open the HTTPS ngrok URL on the phone

## Important note about camera access

For real mobile camera testing, the best option is:
- use a real phone
- open the HTTPS ngrok URL on that phone

Desktop browser DevTools mobile mode is only an emulator. It does not truly turn the desktop into a phone.

## Files your teammates must receive

- `app.py`
- `requirements.txt`
- `install_app.bat`
- `run_app.bat`
- `start_ngrok.bat`
- `APP_ARCHITECTURE/`
- `MODEL/`
- `Branding/`

## Optional

If a teammate already has a terminal open, they can also use:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
