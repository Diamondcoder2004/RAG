### Сборка окружения

pip install -r requirements.txt

### Будут проблемы с torch cuda тут уж сам как нибудь

### Llama index
Я решил выбрать llama index вместо langchain потому что он проще и лучше совместим


### document_groups 
Это папка с группированными md файлами для chunking

## TODO Что нужно сделать прежде чем запускать benchmark.py

Нужна векторная база для хранения чанков, рекомендую qdrant в docker запустить

Нужно написать для каждой группы документов свою стратегию чанкинга, обработки это нужно реализовывать в rag/chunking
Дальше есть rag/preprocessing тут написан скрипт для удаления ссылок на изображения в markdown и прочего мусора в тексте
answer_generator.py - это скрипт с классами для взаимодействия с LLM, тут же промт и вызов Ollama на твоем хосте
retriver Тут классы для поиска по базе с использованием embedding модели и reranker 
benchmark.py тут будет тестирование системы.
Формально тут два этапа работы. Первое это chunking и загрузка в бд, вторая это уже benchmarking так сказать
### Рекомендуемый baseline 
LLM -- qwen2.5:3b
embedder -- sentence-transformers__paraphrase-multilingual-mpnet-base-v2
reranker -- BAAI__bge-reranker-base
Хорошо будет протестировать следующее:
1. GigaEmbeddings, 
3. Более качественную LLM
4. BAAI__bge-reranker-large или что то другое
   
Вообще можно тестировать из различных бенчмарков модели, например отсюда https://huggingface.co/spaces/Vikhrmodels/arenahardlb - бенчмарк LLM для русского языка

#### Доп инфа


Инфа из сайта Россети
Дополнительный faq вопросы: https://портал-тп.рф/platform/portal/tehprisEE_newConnection
Документы правовые	https://www.rosseti.ru/consumers/consumers-of-subsidiaries-and-affiliates/regulatory-framework/

Инфа из СУЕНКО 
Дополнительный faq вопросы:  https://suenco.ru/klientam/obratnaja-svjaz/
