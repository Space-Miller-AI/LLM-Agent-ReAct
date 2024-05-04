from unpywall import Unpywall

from unpywall.utils import UnpywallCredentials

UnpywallCredentials('lorenczhuka28@gmail.com')


res = Unpywall.query(query='sea lion',
               is_oa=True)


print(res.iloc[0]['best_oa_location.url'])
print(res.iloc[0]['best_oa_location.url_for_pdf'])
