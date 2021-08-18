var droneIcon = L.icon({
        iconUrl: 'drone.png',

        iconSize: [60, 60], // size of the icon
        iconAnchor: [0, 0], // point of the icon which will correspond to marker's location
        popupAnchor: [-3, -76] // point from which the popup should open relative to the iconAnchor
    }),
    trail_drone = {
        type: 'Feature',
        properties: {
            id: "drone_line",
            color: "#000055"
        },
        geometry: {
            type: 'LineString',
            coordinates: []
        }
    },
    trail_human = {
        type: 'Feature',
        properties: {
            id: "human_line",
            color: "red"
        },
        geometry: {
            type: 'LineString',
            coordinates: []
        },
    };



var map = L.map('map', {
    layers: [
        L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        })
    ]
});

var result = {
    type: 'FeatureCollection',
    features: [],
}



gps_url = 'http://127.0.0.1:5000/'
realtime = L.realtime(function(success, error) {
    fetch(gps_url)
        .then(function(response) { return response.json(); })
        .then(function(data) {
            var trailCoords_drone = trail_drone.geometry.coordinates;
            trailCoords_drone.push(data.drone.geometry.coordinates);
            trailCoords_drone.splice(0, Math.max(0, trailCoords_drone.length - 5));

            var trailCoords_human = trail_human.geometry.coordinates;
            trailCoords_human.push(data.human.geometry.coordinates);
            trailCoords_human.splice(0, Math.max(0, trailCoords_human.length - 5));
            result = {
                type: 'FeatureCollection',
                features: [data.drone, data.human, trail_drone, trail_human],
            };
            success(result);
        })
        .catch(error);
}, { interval: 250 }).addTo(map);

var geoJsonLayer = L.geoJson(result, {
    onEachFeature: function(feature, layer) {
        if (layer instanceof L.Polyline) {
            layer.setStyle({
                'color': feature.properties.color
            });
        }
    }
}).addTo(map);

realtime.on('update', function(e) {
    if ((this._requestCount) < 10) {
        map.fitBounds(realtime.getBounds(), { maxZoom: 100 });
    } else {}
    Object.keys(e.update).forEach(function(did) {
        var feature = e.update[did];
        // this.getLayer(id).bindPopup("working!");
        this.getLayer(did).bindPopup(feature.properties.id + "<br>" + feature.geometry.coordinates[0] + "<br>" + feature.geometry.coordinates[1]);
        //this.getLayer(id).bindPopup(feature.geometry.coordinates[0] + "<br>" + feature.geometry.coordinates[1]);
    }.bind(this));
});