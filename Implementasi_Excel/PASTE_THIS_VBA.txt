Option Explicit

' =========================================================================
' Modul VBA: Simulasi Step-by-Step Sesuai Metodologi Skripsi
' =========================================================================

' 1. Case Folding (Mengubah teks menjadi huruf kecil)
Function VBA_CaseFolding(ByVal text As String) As String
    VBA_CaseFolding = LCase(text)
End Function

' 1.5. Normalization (Memperbaiki singkatan / slang words secara lengkap)
Function VBA_Normalization(ByVal text As String) As String
    ' Mengganti singkatan umum menjadi bentuk yang tidak terpisah oleh tanda baca
    
    ' Kelompok 1: Not
    text = Replace(text, "can't", "can not")
    text = Replace(text, "won't", "will not")
    text = Replace(text, "don't", "do not")
    text = Replace(text, "didn't", "did not")
    text = Replace(text, "doesn't", "does not")
    text = Replace(text, "isn't", "is not")
    text = Replace(text, "aren't", "are not")
    text = Replace(text, "wasn't", "was not")
    text = Replace(text, "weren't", "were not")
    text = Replace(text, "haven't", "have not")
    text = Replace(text, "hasn't", "has not")
    text = Replace(text, "hadn't", "had not")
    text = Replace(text, "couldn't", "could not")
    text = Replace(text, "shouldn't", "should not")
    text = Replace(text, "wouldn't", "would not")
    text = Replace(text, "mustn't", "must not")
    
    ' Kelompok 2: Am / Is / Are
    text = Replace(text, "i'm", "i am")
    text = Replace(text, "it's", "it is")
    text = Replace(text, "he's", "he is")
    text = Replace(text, "she's", "she is")
    text = Replace(text, "that's", "that is")
    text = Replace(text, "who's", "who is")
    text = Replace(text, "what's", "what is")
    text = Replace(text, "you're", "you are")
    text = Replace(text, "we're", "we are")
    text = Replace(text, "they're", "they are")
    
    ' Kelompok 3: Will / Would / Have
    text = Replace(text, "i'll", "i will")
    text = Replace(text, "you'll", "you will")
    text = Replace(text, "he'll", "he will")
    text = Replace(text, "she'll", "she will")
    text = Replace(text, "it'll", "it will")
    text = Replace(text, "we'll", "we will")
    text = Replace(text, "they'll", "they will")
    text = Replace(text, "i'd", "i would")
    text = Replace(text, "you'd", "you would")
    text = Replace(text, "he'd", "he would")
    text = Replace(text, "she'd", "she would")
    text = Replace(text, "we'd", "we would")
    text = Replace(text, "they'd", "they would")
    text = Replace(text, "i've", "i have")
    text = Replace(text, "you've", "you have")
    text = Replace(text, "we've", "we have")
    text = Replace(text, "they've", "they have")
    
    ' Langkah Sapu Jagat: Hapus apostrof (') yang tersisa
    text = Replace(text, "'", "")
    
    VBA_Normalization = text
End Function

' 2. Remove Noise (Menghapus angka, tanda baca, URL, emoji)
Function VBA_RemoveNoise(ByVal text As String) As String
    Dim i As Integer
    Dim result As String
    Dim c As String
    
    result = ""
    For i = 1 To Len(text)
        c = Mid(text, i, 1)
        ' Hanya pertahankan huruf (a-z), angka (0-9), dan spasi.
        ' Seluruh simbol dan emoji akan otomatis jatuh ke 'Else' dan terhapus menjadi spasi.
        If (AscW(c) >= 97 And AscW(c) <= 122) Or (AscW(c) >= 48 And AscW(c) <= 57) Or c = " " Then
            result = result & c
        Else
            result = result & " "
        End If
    Next i
    
    ' Hapus spasi ganda
    Do While InStr(result, "  ") > 0
        result = Replace(result, "  ", " ")
    Loop
    
    VBA_RemoveNoise = Trim(result)
End Function

' 3. Tokenization (Memotong kalimat menjadi deretan kata)
Function VBA_Tokenize(ByVal text As String) As String
    Dim words() As String
    Dim i As Integer
    Dim result As String
    
    words = Split(text, " ")
    result = "["
    For i = LBound(words) To UBound(words)
        If Len(words(i)) > 0 Then
            result = result & "'" & words(i) & "', "
        End If
    Next i
    
    If Len(result) > 2 Then
        result = Left(result, Len(result) - 2)
    End If
    result = result & "]"
    
    VBA_Tokenize = result
End Function

' Helper untuk format Array String Python
Function CleanArrayString(ByVal arrStr As String) As String
    arrStr = Replace(arrStr, "[", "")
    arrStr = Replace(arrStr, "]", "")
    arrStr = Replace(arrStr, "'", "")
    arrStr = Replace(arrStr, " ", "")
    CleanArrayString = arrStr
End Function

' 4. Stopword Removal (Menghapus kata umum)
Function VBA_RemoveStopwords(ByVal textArray As String) As String
    Dim words() As String
    Dim stoplist As Variant
    Dim i As Integer, j As Integer
    Dim result As String
    Dim isStopword As Boolean
    
    textArray = CleanArrayString(textArray)
    words = Split(textArray, ",")
    
    stoplist = Array("i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", _
                     "it", "its", "they", "them", "their", "what", "which", "who", "this", "that", _
                     "am", "is", "are", "was", "were", "be", "been", "have", "has", "had", _
                     "do", "does", "did", "a", "an", "the", "and", "but", "if", "or", "because", _
                     "as", "until", "while", "of", "at", "by", "for", "with", "about", "to", "from")
                     
    result = "["
    For i = LBound(words) To UBound(words)
        isStopword = False
        For j = LBound(stoplist) To UBound(stoplist)
            If words(i) = stoplist(j) Then
                isStopword = True
                Exit For
            End If
        Next j
        
        ' Cek apakah kata tersebut adalah stopword ATAU apakah kata tersebut murni angka (numeric)
        If Not isStopword And Not IsNumeric(words(i)) And Len(words(i)) > 0 Then
            result = result & "'" & words(i) & "', "
        End If
    Next i
    
    If Len(result) > 2 Then result = Left(result, Len(result) - 2)
    result = result & "]"
    VBA_RemoveStopwords = result
End Function

' 5. Stemming (Mengubah kata menjadi kata dasar)
Function VBA_Stemming(ByVal textArray As String) As String
    Dim words() As String
    Dim i As Integer
    Dim result As String
    Dim word As String
    
    textArray = CleanArrayString(textArray)
    words = Split(textArray, ",")
    
    result = "["
    For i = LBound(words) To UBound(words)
        word = words(i)
        ' Simple Suffix Stripping
        If Len(word) > 4 Then
            If Right(word, 3) = "ing" Then word = Left(word, Len(word) - 3)
            If Right(word, 2) = "ed" Then word = Left(word, Len(word) - 2)
        End If
        If Len(word) > 3 Then
            If Right(word, 1) = "s" And Right(word, 2) <> "ss" Then word = Left(word, Len(word) - 1)
        End If
        
        If Len(word) > 0 Then result = result & "'" & word & "', "
    Next i
    
    If Len(result) > 2 Then result = Left(result, Len(result) - 2)
    result = result & "]"
    VBA_Stemming = result
End Function

' 6. Ekstraksi Fitur TF-IDF (Dibuat spesifik untuk menghindari error koma Excel)
Function Core_TFIDF(ByVal textArray As String, ByVal targetWord As String, ByVal idf As Double) As Double
    Dim words() As String
    Dim wordCount As Integer, matchCount As Integer, i As Integer
    Dim tf As Double
    
    textArray = CleanArrayString(textArray)
    words = Split(textArray, ",")
    wordCount = UBound(words) - LBound(words) + 1
    
    If wordCount = 0 Or textArray = "" Then
        Core_TFIDF = 0
        Exit Function
    End If
    
    matchCount = 0
    For i = LBound(words) To UBound(words)
        If Trim(words(i)) = targetWord Then matchCount = matchCount + 1
    Next i
    
    tf = matchCount / wordCount
    Core_TFIDF = Round(tf * idf, 4)
End Function

Function VBA_TFIDF_terribl(ByVal textArray As String) As Double
    ' Menggunakan bobot IDF negatif (-2.5) untuk kata negatif agar nilainya menjadi minus
    VBA_TFIDF_terribl = Core_TFIDF(textArray, "terrible", -2.5)
End Function

Function VBA_TFIDF_bad(ByVal textArray As String) As Double
    ' Menggunakan bobot IDF negatif (-2.1) untuk kata negatif agar nilainya menjadi minus
    VBA_TFIDF_bad = Core_TFIDF(textArray, "bad", -2.1)
End Function

Function VBA_TFIDF_good(ByVal textArray As String) As Double
    VBA_TFIDF_good = Core_TFIDF(textArray, "good", 1.5)
End Function

Function VBA_TFIDF_love(ByVal textArray As String) As Double
    VBA_TFIDF_love = Core_TFIDF(textArray, "love", 1.8)
End Function

Function VBA_TFIDF_worst(ByVal textArray As String) As Double
    ' Negatif tambahan 1
    VBA_TFIDF_worst = Core_TFIDF(textArray, "worst", -2.8)
End Function

Function VBA_TFIDF_hate(ByVal textArray As String) As Double
    ' Negatif tambahan 2
    VBA_TFIDF_hate = Core_TFIDF(textArray, "hate", -2.4)
End Function

Function VBA_TFIDF_great(ByVal textArray As String) As Double
    ' Positif tambahan 1
    VBA_TFIDF_great = Core_TFIDF(textArray, "great", 2.2)
End Function

Function VBA_TFIDF_excellent(ByVal textArray As String) As Double
    ' Positif tambahan 2
    VBA_TFIDF_excellent = Core_TFIDF(textArray, "excellent", 2.5)
End Function

' 7. Klasifikasi Model Random Forest
Function VBA_Predict_RF(features As Range) As String
    Dim f1 As Double, f2 As Double, f3 As Double, f4 As Double
    Dim f5 As Double, f6 As Double, f7 As Double, f8 As Double
    Dim totalScore As Integer
    On Error GoTo ErrorHandler
    
    f1 = features.Cells(1, 1).Value
    f2 = features.Cells(1, 2).Value
    f3 = features.Cells(1, 3).Value
    f4 = features.Cells(1, 4).Value
    f5 = features.Cells(1, 5).Value
    f6 = features.Cells(1, 6).Value
    f7 = features.Cells(1, 7).Value
    f8 = features.Cells(1, 8).Value
    
    ' Jika semua nilai TF-IDF 0, maka kembalikan neutral
    If f1 = 0 And f2 = 0 And f3 = 0 And f4 = 0 And f5 = 0 And f6 = 0 And f7 = 0 And f8 = 0 Then
        VBA_Predict_RF = "neutral"
        Exit Function
    End If
    
    totalScore = 0
    
    ' f1 (terrible), f2 (bad), f5 (worst), f6 (hate) menghasilkan nilai minus, cek jika < 0
    If f1 < 0 Then totalScore = totalScore - 1
    If f2 < 0 Then totalScore = totalScore - 1
    If f5 < 0 Then totalScore = totalScore - 1
    If f6 < 0 Then totalScore = totalScore - 1
    
    ' f3 (good), f4 (love), f7 (great), f8 (excellent) adalah sentimen positif
    If f3 > 0 Then totalScore = totalScore + 1
    If f4 > 0 Then totalScore = totalScore + 1
    If f7 > 0 Then totalScore = totalScore + 1
    If f8 > 0 Then totalScore = totalScore + 1
    
    If totalScore > 0 Then 
        VBA_Predict_RF = "positive" 
    ElseIf totalScore < 0 Then 
        VBA_Predict_RF = "negative"
    Else
        VBA_Predict_RF = "neutral"
    End If
    
    Exit Function
ErrorHandler:
    VBA_Predict_RF = "Error"
End Function

' 8. Klasifikasi Model SVM (Support Vector Machine)
Function VBA_Predict_SVM(features As Range) As String
    Dim f1 As Double, f2 As Double, f3 As Double, f4 As Double
    Dim f5 As Double, f6 As Double, f7 As Double, f8 As Double
    Dim w1 As Double, w2 As Double, w3 As Double, w4 As Double
    Dim w5 As Double, w6 As Double, w7 As Double, w8 As Double
    Dim bias As Double, sum As Double
    On Error GoTo ErrorHandler
    
    f1 = features.Cells(1, 1).Value
    f2 = features.Cells(1, 2).Value
    f3 = features.Cells(1, 3).Value
    f4 = features.Cells(1, 4).Value
    f5 = features.Cells(1, 5).Value
    f6 = features.Cells(1, 6).Value
    f7 = features.Cells(1, 7).Value
    f8 = features.Cells(1, 8).Value
    
    ' Jika semua fitur 0, langsung neutral
    If f1 = 0 And f2 = 0 And f3 = 0 And f4 = 0 And f5 = 0 And f6 = 0 And f7 = 0 And f8 = 0 Then
        VBA_Predict_SVM = "neutral"
        Exit Function
    End If
    
    ' Bobot hiperbidang (hyperplane weights) simulasi
    ' Karena fitur negatif berharga minus, bobotnya positif (agar sum menjadi negatif)
    w1 = 1.5: w2 = 1.2: w3 = 1.1: w4 = 1.3
    w5 = 1.8: w6 = 1.4: w7 = 1.6: w8 = 1.5
    bias = 0.1
    
    sum = (f1 * w1) + (f2 * w2) + (f3 * w3) + (f4 * w4) + (f5 * w5) + (f6 * w6) + (f7 * w7) + (f8 * w8) + bias
    
    If sum > 0.2 Then 
        VBA_Predict_SVM = "positive" 
    ElseIf sum < -0.2 Then 
        VBA_Predict_SVM = "negative"
    Else
        VBA_Predict_SVM = "neutral"
    End If
    
    Exit Function
ErrorHandler:
    VBA_Predict_SVM = "Error"
End Function

' 9. Behavioral CLV (Menghitung Lifespan)
Function VBA_CLV_Lifespan(ByVal sentiment As String) As Integer
    If LCase(sentiment) = "positive" Then
        VBA_CLV_Lifespan = 12
    ElseIf LCase(sentiment) = "neutral" Then
        VBA_CLV_Lifespan = 6
    Else
        VBA_CLV_Lifespan = 1
    End If
End Function

' 10. Behavioral CLV (Menghitung Potential Loss untuk asumsi paket Rp 186.000)
Function VBA_CLV_Loss(ByVal lifespan As Integer) As Currency
    Dim lost_months As Integer
    lost_months = 12 - lifespan
    VBA_CLV_Loss = lost_months * 186000
End Function
