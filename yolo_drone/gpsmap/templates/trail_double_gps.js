var map = L.map('map', {
        layers: [
            L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
            })
        ]
    }),
    trail_drone = {
        type: 'Feature',
        geometry: {
            type: 'LineString',
            coordinates: []
        },
        properties: {
            id: "drone_line"
        },
    },

    trail_human = {
        type: 'Feature',
        geometry: {
            type: 'LineString',
            coordinates: []
        },
        properties: {
            id: "human_line"
        },
    },

    droneIcon = L.icon({
        iconUrl: 'drone.png',

        iconSize: [60, 60], // size of the icon
        iconAnchor: [0, 0], // point of the icon which will correspond to marker's location
        popupAnchor: [-3, -76] // point from which the popup should open relative to the iconAnchor
    });

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
            success({
                type: 'FeatureCollection',
                features: [data.drone, trail_drone, data.human, trail_human],

            });
        })
        .catch(error);
}, {
    interval: 250
}).addTo(map);

realtime.on('update', function(e) {
    if ((this._requestCount) < 3) {
        map.fitBounds(realtime.getBounds(), { maxZoom: 100 });
    } else {}
    Object.keys(e.update).forEach(function(id) {
        var feature = e.update[id];
        // this.getLayer(id).bindPopup("working!");
        this.getLayer(id).bindPopup(feature.properties.id + "<br>" + feature.geometry.coordinates[0] + "<br>" + feature.geometry.coordinates[1]);
        //this.getLayer(id).bindPopup(feature.geometry.coordinates[0] + "<br>" + feature.geometry.coordinates[1]);
    }.bind(this));
});