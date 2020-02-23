from geopy.geocoders import Nominatim

print("Please enter address");
address=input();
geolocator = Nominatim(user_agent="specify_your_app_name_here")
location = geolocator.geocode(address)
print((location.latitude, location.longitude))

