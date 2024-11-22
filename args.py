# 실행 인자 관리 함수
def parse_args():
    """
    학습에 필요한 하이퍼파라미터 설정
    
    Returns:
        args: 설정된 인자들을 포함하는 namespace 객체
    """
    parser = argparse.ArgumentParser(description='모델 학습을 위한 인자 설정')
    
    # 모델 관련 인자
    parser.add_argument('--encoding_dim', type=int, default=32, 
                        help='인코더의 출력 차원')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='분류할 클래스 수')
    
    # 학습 관련 인자 
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='드롭아웃 비율')

    # 데이터 관련 인자
    parser.add_argument('--data_path', type=str, default='./Data',
                        help='데이터 경로')
    
    args = parser.parse_args()
    return args