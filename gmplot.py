# import gmplot package
import gmplot
import webbrowser, os

#Set different latitude and longitude points
latitude1, longitude1 = zip(*[
   (17.3950, 78.3968),(17.3987, 78.2988),(17.3956, 78.4750)])
#declare the center of the map, and how much we want the map zoomed in
gmap3 = gmplot.GoogleMapPlotter(latitude1[0], longitude1[0], 13)
# Scatter map
gmap3.scatter( latitude1, longitude1, '#FF0000',size = 50, marker = False )
# Plot method Draw a line in between given coordinates

#Your Google_API_Key
gmap3.apikey = 'AIzaSyDmllc9JkG8RRgTiriuzx-mwIziEyrzB7c'
# save it to html
gmap3.draw(r"C:\\Users\\asis\\Desktop\\map11.html")
webbrowser.open("C:\\Users\\asis\\Desktop\\map11.html")