Attribute VB_Name = "Module_NLP_RF"
Option Explicit

' =========================================================================
' Modul VBA: Simulasi Tokenisasi, NLTK Stopwords, & Random Forest
' =========================================================================

' Fungsi: Menghapus Tanda Baca
Function RemovePunctuation(ByVal text As String) As String
    Dim i As Integer
    Dim result As String
    Dim c As String
    
    result = ""
    For i = 1 To Len(text)
        c = Mid(text, i, 1)
        ' Hanya pertahankan huruf dan spasi
        If (Asc(LCase(c)) >= 97 And Asc(LCase(c)) <= 122) Or c = " " Then
            result = result & c
        Else
            result = result & " "
        End If
    Next i
    
    ' Hapus spasi ganda
    Do While InStr(result, "  ") > 0
        result = Replace(result, "  ", " ")
    Loop
    
    RemovePunctuation = Trim(result)
End Function

' Fungsi: Menghapus Stopwords & Format Tokenize Array
Function NLTK_Preprocess(ByVal text As String) As String
    Dim words() As String
    Dim stoplist As Variant
    Dim i As Integer, j As Integer
    Dim result As String
    Dim isStopword As Boolean
    
    ' Bersihkan tanda baca dan emoji terlebih dahulu
    text = RemovePunctuation(text)
    
    ' Daftar stopword sederhana
    stoplist = Array("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", _
                     "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", _
                     "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", _
                     "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", _
                     "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", _
                     "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", _
                     "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", _
                     "about", "against", "between", "into", "through", "during", "before", "after", _
                     "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", _
                     "under", "again", "further", "then", "once", "here", "there", "when", "where", _
                     "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", _
                     "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", _
                     "very", "s", "t", "can", "will", "just", "don", "should", "now")
                     
    words = Split(LCase(text), " ")
    result = "["
    
    For i = LBound(words) To UBound(words)
        isStopword = False
        For j = LBound(stoplist) To UBound(stoplist)
            If words(i) = stoplist(j) Then
                isStopword = True
                Exit For
            End If
        Next j
        
        ' Lakukan Simple Stemming (Hapus -ing, -ed, -s)
        Dim word As String
        word = words(i)
        
        If Not isStopword And Len(word) > 1 Then
            If Right(word, 3) = "ing" Then word = Left(word, Len(word) - 3)
            If Right(word, 2) = "ed" Then word = Left(word, Len(word) - 2)
            If Right(word, 1) = "s" And Right(word, 2) <> "ss" Then word = Left(word, Len(word) - 1)
            
            result = result & "'" & word & "', "
        End If
    Next i
    
    If Len(result) > 2 Then
        result = Left(result, Len(result) - 2)
    End If
    result = result & "]"
    
    NLTK_Preprocess = result
End Function

' =========================================================================
' Fungsi: Prediksi Random Forest Classifier Sederhana
' =========================================================================
Function Predict_RandomForest(features As Range) As String
    ' Menerima Range (contoh D2:G2) berupa Angka TF-IDF dan memprediksi sentimen
    
    Dim f1 As Double, f2 As Double, f3 As Double, f4 As Double
    Dim score1 As Integer, score2 As Integer, score3 As Integer
    Dim votePos As Integer, voteNeg As Integer
    
    On Error GoTo ErrorHandler
    
    ' Mengambil nilai TF-IDF dari Range
    f1 = Val(features.Cells(1, 1).Value)
    f2 = Val(features.Cells(1, 2).Value)
    f3 = Val(features.Cells(1, 3).Value)
    f4 = Val(features.Cells(1, 4).Value)
    
    votePos = 0
    voteNeg = 0
    
    ' Pohon Keputusan 1
    If f1 > 0.5 Then 
        score1 = 1
    Else
        score1 = -1
    End If
    
    ' Pohon Keputusan 2
    If f2 > 0.3 Then 
        score2 = -1
    Else
        score2 = 1
    End If
    
    ' Pohon Keputusan 3
    If f3 > 0.1 And f4 < 0.2 Then
        score3 = 1
    Else
        score3 = -1
    End If
    
    ' Voting (Ensemble)
    If score1 = 1 Then votePos = votePos + 1 Else voteNeg = voteNeg + 1
    If score2 = 1 Then votePos = votePos + 1 Else voteNeg = voteNeg + 1
    If score3 = 1 Then votePos = votePos + 1 Else voteNeg = voteNeg + 1
    
    If votePos > voteNeg Then
        Predict_RandomForest = "positive"
    Else
        Predict_RandomForest = "negative"
    End If
    Exit Function

ErrorHandler:
    Predict_RandomForest = "Error"
End Function

' =========================================================================
' Fungsi: Simulasi Kalkulasi Bobot TF-IDF Sederhana
' =========================================================================
Function Calculate_TFIDF(ByVal textArray As String, ByVal targetWord As String) As Double
    ' Menghitung TF (Term Frequency) secara sederhana dari string array NLTK
    ' Dan mengalikan dengan IDF statis (sebagai simulasi)
    
    Dim words() As String
    Dim wordCount As Integer
    Dim matchCount As Integer
    Dim i As Integer
    Dim tf As Double
    Dim idf As Double
    
    ' Bersihkan format array python (tanda kurung siku dan kutip)
    textArray = Replace(textArray, "[", "")
    textArray = Replace(textArray, "]", "")
    textArray = Replace(textArray, "'", "")
    textArray = Replace(textArray, " ", "")
    
    words = Split(LCase(textArray), ",")
    wordCount = UBound(words) - LBound(words) + 1
    
    If wordCount = 0 Then
        Calculate_TFIDF = 0
        Exit Function
    End If
    
    matchCount = 0
    For i = LBound(words) To UBound(words)
        If Trim(words(i)) = LCase(targetWord) Then
            matchCount = matchCount + 1
        End If
    Next i
    
    ' Menghitung Term Frequency (TF)
    tf = matchCount / wordCount
    
    ' Simulasi Inverse Document Frequency (IDF) untuk kata-kata kunci
    ' Pada praktiknya IDF dihitung dari log(TotalDokumen / DokumenMengandungKata)
    Select Case LCase(targetWord)
        Case "terribl"
            idf = 2.5
        Case "bad"
            idf = 2.1
        Case "good"
            idf = 1.5
        Case "love"
            idf = 1.8
        Case Else
            idf = 1.2
    End Select
    
    ' Rumus Akhir TF-IDF
    Calculate_TFIDF = Round(tf * idf, 4)
End Function
