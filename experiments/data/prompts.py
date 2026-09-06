ANSWER_KEYS = {
    "Bengali": "উত্তর",
    "Mandarin Chinese (Standard Beijing)": "答案",
    "German": "Antwort",
    "French": "Réponse",
    "Japanese": "答え",
    "Russian": "Ответ",
    "Spanish (Latin American)": "Respuesta",
    "Telugu": "సమాధానం",
    "Thai": "คำตอบ",
    "Swahili": "Jibu",
    "Greek": "Απάντηση",
    "Hebrew": "תשובה",
    "Korean": "정답",
    "Vietnamese": "Câu trả lời",
}

GMMLU_PROMPTS = {
    'English':
[
'''Answer the following multiple choice question.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

    'Mandarin Chinese (Standard Beijing)':
[
    '''回答以下选择题。
你的回答最后一行必须满足以下格式：'Answer: $LETTER'（不含引号），其中 LETTER 为 A/B/C/D 之一。

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Arabic':
[
    '''أجب عن سؤال الاختيار من متعدد التالي.
يجب أن تكون السطر الأخير من إجابتك بالشكل التالي:
'Answer: $LETTER'
(من دون علامات اقتباس)، حيث LETTER هو أحد الحروف A أو B أو C أو D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Bengali':
[
    '''নিম্নলিখিত বহু নির্বাচনী প্রশ্নের উত্তর দিন।
আপনার উত্তরের শেষ লাইনটি অবশ্যই এই ফরম্যাটে হতে হবে:
'Answer: $LETTER'
(উদ্ধৃতি চিহ্ন ছাড়া), যেখানে LETTER হলো A, B, C, অথবা D-এর একটি।

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'German':
[
    '''Beantworte die folgende Multiple-Choice-Frage.
Die letzte Zeile deiner Antwort muss folgendes Format haben:
'Answer: $LETTER'
(ohne Anführungszeichen), wobei LETTER eines der Zeichen A, B, C oder D ist.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Spanish (Latin American)':
[
    '''Responde la siguiente pregunta de opción múltiple.
La última línea de tu respuesta debe tener el siguiente formato:
'Answer: $LETTER'
(sin comillas), donde LETTER es una de las letras A, B, C o D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'French':
[
    '''Répondez à la question à choix multiple suivante.
La dernière ligne de votre réponse doit être au format suivant :
'Answer: $LETTER'
(sans guillemets), où LETTER est l’une des lettres A, B, C ou D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Hindi':
[
    '''निम्नलिखित बहुविकल्पीय प्रश्न का उत्तर दें।
आपके उत्तर की अंतिम पंक्ति निम्नलिखित प्रारूप में होनी चाहिए:
'Answer: $LETTER'
(बिना उद्धरण चिह्नों के), जहाँ LETTER A, B, C या D में से एक है।

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Indonesian':
[
    '''Jawablah pertanyaan pilihan ganda berikut.
Baris terakhir dari jawaban Anda harus menggunakan format berikut:
'Answer: $LETTER'
(tanpa tanda kutip), di mana LETTER adalah salah satu dari A, B, C, atau D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Italian':
[
    '''Rispondi alla seguente domanda a scelta multipla.
L’ultima riga della tua risposta deve essere nel seguente formato:
'Answer: $LETTER'
(senza virgolette), dove LETTER è una delle lettere A, B, C o D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Japanese':
[
    '''次の選択式の質問に答えてください。
回答の最後の行は、以下の形式にしてください：
'Answer: $LETTER'
（引用符なし）、ここで LETTER は A, B, C, D のいずれかです。

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Korean':
[
    '''다음 객관식 질문에 답하세요.
답변의 마지막 줄은 다음 형식을 따라야 합니다:
'Answer: $LETTER'
(따옴표 없이), 여기서 LETTER는 A, B, C, D 중 하나입니다.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Portuguese (Brazilian)':
[
    '''Responda à seguinte pergunta de múltipla escolha.
A última linha da sua resposta deve estar no seguinte formato:
'Answer: $LETTER'
(sem aspas), onde LETTER é uma das letras A, B, C ou D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Swahili':
[
    '''Jibu swali linalofuata la chaguo nyingi.
Mstari wa mwisho wa jibu lako lazima uwe katika muundo huu:
'Answer: $LETTER'
(bila alama za nukuu), ambapo LETTER ni moja kati ya A, B, C, au D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Yoruba':
[
    '''Dáhùn ìbéèrè yìí tí ó ní àṣàyàn ọ̀pọ̀.
Ìlà tí ó kẹ́yìn nínú ìdáhùn rẹ gbọ́dọ̀ jẹ́ ní ìṣàkóso yìí:
'Answer: $LETTER'
(láìsí àmì ìsọ̀rọ̀), níbi tí LETTER jẹ́ ọkan lára A, B, C, D.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Russian':
[
    '''Ответьте на следующий вопрос с выбором одного варианта.
Последняя строка вашего ответа должна иметь следующий формат: ‘Answer: $LETTER’ (без кавычек), где LETTER — один из вариантов ABCD.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Telugu':
[
    '''’క్రింది బహుళ ఎంపిక ప్రశ్నకు సమాధానం ఇవ్వండి.
మీ సమాధానంలోని చివరి పంక్తి ఈ క్రింది ఆకృతిలో ఉండాలి: ‘Answer: $LETTER’ (ఉద్ధరణ చిహ్నాలు లేకుండా), ఇక్కడ LETTER అనేది ABCD లోని ఒకటి.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Hebrew':
[
    '''ענה על שאלת הבחירה המרובה הבאה.
השורה האחרונה של התשובה שלך צריכה להיות בפורמט הבא: ‘Answer: $LETTER’ (ללא מרכאות), כאשר LETTER הוא אחד מהאותיות ABCD.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],

'Vietnamese':
[
    '''Hãy trả lời câu hỏi trắc nghiệm sau.
Dòng cuối cùng của câu trả lời của bạn phải có định dạng như sau: 'Answer: $LETTER' (không bao gồm dấu ngoặc kép), trong đó LETTER là một trong các chữ cái ABCD.

{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
],
}


MMLU_QUESTION_PART = [
    '''{question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}''',
]




BELEBELE_PROMPTS = {
    'English':
[
    '''Given the following passage, query, and answer choices, output the letter corresponding to the correct answer.
### Passage:
{passage}

### Query:
{question}

### Choices:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],
    'French':
[
    '''Étant donné le passage suivant, la question et les choix de réponses, indiquez la lettre correspondant à la bonne réponse.
### Passage:
{passage}

### Question:
{question}

### Choix:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Spanish (Latin American)':
[
    '''Dado el siguiente pasaje, la pregunta y las opciones de respuesta, indique la letra correspondiente a la respuesta correcta.
### Pasaje:
{passage}

### Pregunta:
{question}

### Opciones:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Mandarin Chinese (Standard Beijing)':
[
    '''根据以下文章、问题和选项，输出对应正确答案的字母。
### 文章：
{passage}

### 问题：
{question}

### 选项：
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Japanese':
[
    '''以下の文章、質問、および選択肢を読み、正しい答えに対応する文字を出力してください。
### 文章：
{passage}

### 質問：
{question}

### 選択肢：
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Korean':
[
    '''다음 지문, 질문, 그리고 선택지를 보고 올바른 정답에 해당하는 문자를 출력하세요.
### 지문:
{passage}

### 질문:
{question}

### 선택지:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Vietnamese':
[
    '''Dựa vào đoạn văn, câu hỏi và các lựa chọn sau, hãy xuất ra chữ cái tương ứng với đáp án đúng.
### Đoạn văn:
{passage}

### Câu hỏi:
{question}

### Lựa chọn:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Hebrew':
[
    '''בהתבסס על הקטע, השאלה ואפשרויות התשובה הבאות, יש לציין את האות המתאימה לתשובה הנכונה.
### קטע:
{passage}

### שאלה:
{question}

### אפשרויות:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Bengali':
[
    '''নিম্নলিখিত অনুচ্ছেদ, প্রশ্ন এবং উত্তর বিকল্পগুলোর ভিত্তিতে সঠিক উত্তরের সাথে সম্পর্কিত অক্ষরটি প্রদান করুন।
### অনুচ্ছেদ:
{passage}

### প্রশ্ন:
{question}

### বিকল্পসমূহ:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],

    'Telugu':
[
    '''క్రింది ప్యాసేజ్, ప్రశ్న మరియు సమాధాన ఎంపికలను ఆధారంగా తీసుకుని, సరైన సమాధానానికి సంబంధించిన అక్షరాన్ని ఇవ్వండి.
### ప్యాసేజ్:
{passage}

### ప్రశ్న:
{question}

### ఎంపికలు:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}''',
],
}


XQUAD_PROMPTS = {
    "English": [
        """Given the following passage and question, find the answer to the question within the passage.
Extract the answer from the passage and copy it exactly as it appears in the passage. Do not write any additional text.

### Passage:
{passage}

### Question:
{question}""",
    ],
    "Spanish (Latin American)": [
        """Dado el siguiente pasaje y la siguiente pregunta, encuentra la respuesta a la pregunta dentro del pasaje.
Extrae la respuesta del pasaje y cópiala exactamente tal como aparece en el pasaje. No escribas ningún texto adicional.

### Pasaje:
{passage}

### Pregunta:
{question}""",
    ],
    "Greek": [
        """Δεδομένου του παρακάτω αποσπάσματος και της ερώτησης, βρες την απάντηση στην ερώτηση μέσα στο απόσπασμα.
Εξήγαγε την απάντηση από το απόσπασμα και αντέγραψέ την ακριβώς όπως εμφανίζεται στο απόσπασμα. Μην γράψεις κανένα επιπλέον κείμενο.

### Απόσπασμα:
{passage}

### Ερώτηση:
{question}""",
    ],
    "Vietnamese": [
        """Cho đoạn văn và câu hỏi dưới đây, hãy tìm câu trả lời cho câu hỏi trong đoạn văn.
Trích xuất câu trả lời từ đoạn văn và sao chép chính xác như trong đoạn văn. Không viết thêm bất kỳ văn bản nào khác.

### Đoạn văn:
{passage}

### Câu hỏi:
{question}""",
    ],
    "Mandarin Chinese (Standard Beijing)": [
        """请根据以下文章和问题，在文章中找到问题的答案。
请从文章中提取答案，并完全按照文章中的原文复制答案。不要输出任何额外内容。

### 文章：
{passage}

### 问题：
{question}""",
    ],
    "Russian": [
        """Используя приведённые ниже текст и вопрос, найдите ответ на вопрос в тексте.
Извлеките ответ из текста и скопируйте его в точности так, как он представлен в тексте. Не добавляйте никакого дополнительного текста.

### Текст:
{passage}

### Вопрос:
{question}""",
    ],
}
