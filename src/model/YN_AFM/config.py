# Configuration File
# 모든 칼럼 이름, 연속형 변수/범주형 변수 목록을 List로 저장한다.
ORIGINAL_FIELDS = ['locationId', 'createdDate', 'is_fch', 
        'is_local', 'rated_count',
       'average_photonum', 'average_rating', 'category_l',
       'user_visit_history']

CONT_FIELDS = ['locationId', 'createdDate',
       'rated_count', 'average_photonum', 'average_rating', 'user_visit_history']

CAT_FIELDS = ['is_fch', 'category_l', 'is_local']  


ALL_FIELDS = CONT_FIELDS + CAT_FIELDS

NUM_FIELD = len(ALL_FIELDS)
NUM_CONT = len(CONT_FIELDS)

# Hyper-parameters for Experiment
BATCH_SIZE = 2560
EMBEDDING_SIZE = 10
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.2

# for Pair-wise Interaction Layer
MASKS = []
for i in range(NUM_FIELD):
    flag = 1 + i

    MASKS.extend([False]*(flag))
    MASKS.extend([True]*(NUM_FIELD - flag))

