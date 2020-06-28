import json
import requests
import cssselect
import lxml.html
from bs4 import BeautifulSoup
import pandas as pd

# 트리바고 호텔 정보 가져오기
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'
    
}
def getGEO(data):
    hotel_id = data['accommodation']['id']
    name = data['accommodation']['name']
    lat = data['accommodation']['geocode']['lat']
    lng = data['accommodation']['geocode']['lng']
    if data['accommodation'].get('address', 'none') == 'none':
        country, addr, locality, poscal = None, None, None, None
    else:
        country = data['accommodation']['address'].get('country', None)
        addr = data['accommodation']['address'].get('street', None)
        locality = data['accommodation']['address'].get('locality', None)
        poscal = data['accommodation']['address'].get('postalCode', None)
    value = [hotel_id, name, lat, lng, country, addr, locality, poscal]
    return value


# 트리바고 호텔 위치 기반 주변 음식점 가져오기
def getPlaces(item_id, lat, lng):
    for i in range(1,5):
        try:
            url = f'https://map.naver.com/v5/api/around-here/place?caller=pcweb&categoryUsageId=app_aroundme_v2&query=DINING_ALL&searchCoord={lng};{lat}&siteSort=0&page={i}&displayCount=20&&lang=ko'
            response = requests.get(url, headers = headers)
            data = json.loads(response.text)
            items = data['result']['place']['list']
            return items
        except:
            return 'end'


# place 영수증 리뷰에 표시된 user와 review 가져오기        
def getUsers(idx):
    reviews_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'
    }
    
    for page in [0,1,2,3]:
        try: # 페이지가 있으면
            url = f'https://store.naver.com/restaurants/detail?entry=plt&id={idx}&tab=receiptReview&tabPage={page}'
            response = requests.get(url, headers = headers)
            root = lxml.html.fromstring(response.text)
            
            nicks = [elem.text_content().strip() for elem in root.cssselect('div.reviewer > a')]
            reviews = [elem.text_content().strip() for elem in root.cssselect('div.review_txt')]
            stars = [elem.attrib['style'] for elem in root.cssselect('span.bg > span.value')]
            herfs = [elem.attrib['href'].split('my/')[1].split('/')[0] for elem in root.cssselect('a.item')]
            
            value = list(map(list, zip(nicks, reviews, stars, herfs)))
            [x.append(idx) for x in value]
            reviews_list.extend(value)
        except: # 페이지가 없으면
            continue

    return reviews_list

url = "https://m.place.naver.com/my/graphql"
headers = {
  'Content-Type': 'application/json'
}

# 각 user의 href을 사용해서 idno 가져오기
def getIDno(href):
    global url
    global headers
    payload = "{\"operationName\":\"UserProfile\",\"variables\":{\"objectId\":\""+href+"\"},\"query\":\"query UserProfile($objectId: String) {\\n  session(objectId: $objectId) {\\n    loggedIn\\n    authKey\\n    naverId\\n    idno\\n    objectId\\n    isParticipant\\n    grade\\n    gradeChangedAt\\n    guest\\n    defaultFolderShareId\\n    __typename\\n  }\\n  profile(objectId: $objectId) {\\n    objectId\\n    idno\\n    nickname\\n    imageUrl\\n    totalReviews\\n    totalReviewImages\\n    __typename\\n  }\\n}\\n\"}"
    response = requests.request("POST", url, headers=headers, data = payload)
    try:
        data = response.json()
        idno = data['data']['profile']['idno']
        nick = data['data']['profile']['nickname']
        count = data['data']['profile']['totalReviews']
        value = [href, idno, nick, count]
        return value
    except:
        return 'end'
    
def getRv(idno, page):
    global url
    global headers
    payload = "{\"operationName\":\"authorReviews\",\"variables\":{\"authorId\":\"" +idno+"\",\"includeMedia\":false,\"page\":"+page+",\"limit\":100},\"query\":\"query authorReviews($authorId: String, $limit: Int, $page: Int, $rating: Float, $includeMedia: Boolean) {\\n  authorReviews(authorId: $authorId, limit: $limit, page: $page, rating: $rating, includeMedia: $includeMedia) {\\n    reviewCount\\n    imageCount\\n    reviews {\\n      rating\\n      body\\n      visitCount\\n      viewCount\\n      thumbnail\\n      status\\n      media {\\n        type\\n        thumbnail\\n        __typename\\n      }\\n      visitedDate {\\n        date\\n        displayDate\\n        displayDateTime\\n        __typename\\n      }\\n      place {\\n        id\\n        name\\n        category\\n        categoryCode\\n        categoryCodes\\n        phone\\n        address\\n        __typename\\n      }\\n      id\\n      __typename\\n    }\\n    __typename\\n  }\\n}\\n\"}"
    response = requests.request("POST", url, headers=headers, data = payload)
    
    if 'Gateway' in response.text:
        return getRv(idno, page)
    elif 'Too Many Requests' in response.text:
        return 'end'
    else:
        data = response.json()
        review = data['data']['authorReviews']['reviews']
        return review
    
def addv(x, idno):
    x['idno'] = idno
    return x
        

# place.address를 사용하여 위경도 구하기

key = 'd1SvFCY9aQyWtAOaEWfOyur2YSCfyJZnKNJQ3FaN'
headers = {
    'X-NCP-APIGW-API-KEY-ID' : 'vasj6ueva4',
    'X-NCP-APIGW-API-KEY' : 'd1SvFCY9aQyWtAOaEWfOyur2YSCfyJZnKNJQ3FaN'
    }

def NVgeocoding(addr):
    url = f'https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={addr}'
    response = requests.get(url, headers = headers)
    try:
        result = json.loads(response.text)
        jibun = result['addresses'][0]['jibunAddress']
        road = result['addresses'][0].get('roadAddress', 'none')
        y = float(result['addresses'][0]['y'])
        x = float(result['addresses'][0]['x'])
        value = [jibun, road, y, x]
    except:
        value = ['error', 'error', 0, 0]
    return value