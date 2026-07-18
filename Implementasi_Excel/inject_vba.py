import win32com.client
import os

def inject():
    print("Membuka Excel...")
    xl = win32com.client.Dispatch("Excel.Application")
    xl.Visible = False
    xl.DisplayAlerts = False
    
    file_path = os.path.abspath("Netflix_StepByStep_Demo.xlsx")
    wb = xl.Workbooks.Open(file_path)
    
    print("Membaca kode VBA...")
    with open("macro_step_by_step.vba", "r", encoding="utf-8") as f:
        vba_code = f.read()
        
    print("Menyuntikkan VBA...")
    try:
        xlmodule = wb.VBProject.VBComponents.Add(1)  # 1 = vbext_ct_StdModule
        xlmodule.CodeModule.AddFromString(vba_code)
        
        save_path = os.path.abspath("Netflix_Otomatis_VBA.xlsm")
        wb.SaveAs(save_path, FileFormat=52) # 52 = xlOpenXMLWorkbookMacroEnabled
        print("Berhasil menyimpan Netflix_Otomatis_VBA.xlsm!")
    except Exception as e:
        print(f"Gagal menyuntikkan VBA (mungkin terblokir oleh setting Trust Center): {e}")
    finally:
        wb.Close(SaveChanges=False)
        xl.Quit()

if __name__ == "__main__":
    inject()
