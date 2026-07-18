import win32com.client
import os

try:
    xl = win32com.client.Dispatch("Excel.Application")
    xl.Visible = False
    xl.DisplayAlerts = False

    wb_path = os.path.abspath("Netflix_Full_Process_V2.xlsx")
    wb = xl.Workbooks.Open(wb_path)
    
    # Check if we can access VBProject
    try:
        proj = wb.VBProject
        print("Success! Can access VBProject")
    except Exception as e:
        print("Failed to access VBProject. Trust access is likely disabled.")
        print(str(e))
        
    wb.Close(False)
    xl.Quit()
except Exception as e:
    print(f"General error: {e}")
