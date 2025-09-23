import random
from typing import List, Dict

def sample_clip_concat(num_samples: int, nproc: int = 1, resolution=(224, 224), seed: int = 0) -> List[Dict]:
    """
    LAION 데이터셋(예: "laion/laion400m-data")에서 이미지-텍스트 샘플을 추출합니다.
    
    반환 형식:
        List[Dict] 형태로, 각 원소는 {"image": PIL.Image, "text": str} 입니다.
        
    인자:
        num_samples (int): 추출할 총 샘플 수.
        nproc (int): 멀티프로세싱에 사용할 프로세스 수 (현재는 단일 프로세스 방식으로 구현됨).
        resolution (tuple): 이미지 리사이즈 크기 (기본값: (224, 224)).
        seed (int): 랜덤 시드.
        
    주의:
        LAION 데이터셋의 키("URL", "TEXT")는 실제 데이터셋에 따라 달라질 수 있으므로 필요시 수정합니다.
    """
    import requests
    from io import BytesIO
    from PIL import Image
    from datasets import load_dataset

    random.seed(seed)
    results = []
    # streaming 모드로 LAION 데이터셋 로드 (split 및 데이터셋 이름은 상황에 맞게 조정)
    dataset = load_dataset("laion/laion400m", split="train", streaming=True)
    dataset_iter = iter(dataset)

    while len(results) < num_samples:
        try:
            sample = next(dataset_iter)
            # LAION 데이터셋의 이미지 URL과 텍스트 필드 (필드명은 데이터셋에 따라 수정)
            url = sample.get("URL") or sample.get("url")
            text = sample.get("caption")
            if url is None or text is None:
                raise

            response = requests.get(url, timeout=5)
            # 이미지 로딩 및 RGB 변환
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = image.resize(resolution)
            results.append({"image": image, "text": text})
        except Exception as e:
            print(e)
            # 오류 발생 시 해당 샘플은 건너뜁니다.
            continue

    return results

def sample_cc12m_concat(num_samples: int, resolution=(224, 224), seed: int = 0) -> List[Dict]:
    """
    Conceptual Captions 12M (CC-12M)에서 이미지-텍스트 샘플을 추출합니다.
    
    반환 형식:
        [{"image": PIL.Image, "text": str}, ...]
        
    인자:
        num_samples (int): 추출할 총 샘플 수.
        resolution (tuple): 이미지 리사이즈 크기.
        seed (int): 랜덤 시드.
    """
    import requests
    from io import BytesIO
    from PIL import Image
    from datasets import load_dataset

    random.seed(seed)
    results = []
    
    # Hugging Face CC-12M 데이터셋 로드 (streaming 사용 권장)
    dataset = load_dataset("google-research-datasets/conceptual_captions", 'unlabeled', split="train", streaming=False)
    dataset_iter = iter(dataset)

    # while len(results) < num_samples:
    #     print(len(results))
    #     try:
    #         sample = next(dataset_iter)
    #         url = sample.get("image_url")
    #         text = sample.get("caption")
    #         if url is None or text is None:
    #             continue

    #         response = requests.get(url, timeout=10)
    #         image = Image.open(BytesIO(response.content)).convert("RGB")
    #         image = image.resize(resolution)
    #         results.append({"image": image, "text": text})
    #     except Exception as e:
    #         print('##########################')
    #         print(e)
    #         continue  # 실패한 샘플은 건너뜀
    
    while len(results) < num_samples:
        print(len(results))
        try:
            sample = next(dataset_iter)
            url = sample.get("image_url")
            text = sample.get("caption")
            if url is None or text is None:
                continue

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue

            try:
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image = image.resize(resolution)
            except Exception as img_error:
                print('Failed to open image:', img_error)
                continue

            results.append({"image": image, "text": text})
        except Exception as e:
            print('##########################')
            print(e)
            continue


    return results
