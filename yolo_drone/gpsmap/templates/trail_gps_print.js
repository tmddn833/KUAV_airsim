var map = L.map('map', {
        layers: [
            L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
            })
        ],
        center: [0, 0],
        zoom: 0
    }),
    trail = {
        type: 'Feature',
        properties: {
            id: 1
        },
        geometry: {
            type: 'LineString',
            coordinates: []
        }
    },

    url = 'http://127.0.0.1:5000/'

realtime = L.realtime(function(success, error) {
    fetch(url)
        .then(function(response) { return response.json(); })
        .then(function(data) {
            var trailCoords = trail.geometry.coordinates;
            trailCoords.push(data.geometry.coordinates);
            trailCoords.splice(0, Math.max(0, trailCoords.length - 5));
            success({
                type: 'FeatureCollection',
                features: [data, trail]
            });
        })
        .catch(error);
}, {
    interval: 250
}).addTo(map);

realtime.on('update', function(e) {
    //map.fitBounds(realtime.getBounds(), {maxZoom: 3});
    Object.keys(e.update).forEach(function(id) {
        var feature = e.update[id];
        // this.getLayer(id).bindPopup("working!");
        this.getLayer(id).bindPopup(feature.geometry.coordinates[0] + "<br>" + feature.geometry.coordinates[1]);
    }.bind(this));
});