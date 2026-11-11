import sys
sys.path.append('.')
from src.deployment.predictor import FakeNewsPredictor
from src.data.reddit_crawler import RedditCrawler

predictor = FakeNewsPredictor()
crawler = RedditCrawler()
crawler._setup_session()

def find_working_real_link():
    url_base = 'https://www.reddit.com/r/technology/hot.json'
    resp = crawler.session.get(url_base, headers={'User-Agent': 'Mozilla/5.0'})
    data = resp.json()
    for child in data['data']['children']:
        p = child['data']
        if p['num_comments'] > 2 and not p.get('is_video') and not p.get('stickied'):
            link = 'https://www.reddit.com' + p['permalink']
            try:
                res = predictor.predict_url(link)
                print(f"{link} -> {res['prediction']} ({res['confidence']:.4f}) Nodes: {res['graph_info']['num_nodes']}")
                if res['prediction'] == 'REAL':
                    print("FOUND WORKING REAL LINK:", link)
                    return
            except Exception as e:
                pass

find_working_real_link()
