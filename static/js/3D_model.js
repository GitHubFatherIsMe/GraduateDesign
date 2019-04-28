var camera, scene, renderer
var controls
var root
var loader = new THREE.PDBLoader()
var offset = new THREE.Vector3()
function init(file_path) {
    scene = new THREE.Scene()
    scene.background = new THREE.Color( 0x050505 )
    camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 1, 5000 )
    camera.position.set(0,0,5000)
    scene.add( camera )
    var light = new THREE.DirectionalLight( 0xffffff, 0.8 )
    light.position.set(1, 1, 1)
    scene.add( light )
    var light = new THREE.DirectionalLight( 0xffffff, 0.5 )
    light.position.set(-1, -1, 1)
    scene.add( light )
    root = new THREE.Group()
    scene.add( root )
    //
    renderer = new THREE.WebGLRenderer( { antialias: true } )
    renderer.setPixelRatio( window.devicePixelRatio )
    renderer.setSize( window.innerWidth/2, window.innerHeight/2 )
    document.getElementById( 'container' ).appendChild( renderer.domElement )
    //
    controls = new THREE.TrackballControls( camera, renderer.domElement )
    controls.minDistance = 500
    controls.maxDistance = 5000
    //
    loadMolecule( file_path )
    //
    window.addEventListener( 'resize', onWindowResize, false )
}

//
function loadMolecule( url ) {
    while ( root.children.length > 0 ) {
        var object = root.children[ 0 ]
        object.parent.remove( object )
    }
    loader.load( url, function ( pdb ) {
        var geometryAtoms = pdb.geometryAtoms
        var geometryBonds = pdb.geometryBonds
        var json = pdb.json
        var boxGeometry = new THREE.BoxBufferGeometry( 1, 1, 1 )
        var sphereGeometry = new THREE.IcosahedronBufferGeometry( 1, 2 )
        geometryAtoms.computeBoundingBox()
        geometryAtoms.boundingBox.getCenter( offset ).negate()
        geometryAtoms.translate( offset.x, offset.y, offset.z )
        geometryBonds.translate( offset.x, offset.y, offset.z )
        var positions = geometryAtoms.getAttribute( 'position' )
        var colors = geometryAtoms.getAttribute( 'color' )
        var position = new THREE.Vector3()
        var color = new THREE.Color()
        for ( var i = 0; i < positions.count; i ++ ) {
            position.x = positions.getX( i )
            position.y = positions.getY( i )
            position.z = positions.getZ( i )
            color.r = colors.getX( i )
            color.g = colors.getY( i )
            color.b = colors.getZ( i )
            var material = new THREE.MeshPhongMaterial( { color: color } )
            var object = new THREE.Mesh( sphereGeometry, material )
            object.position.copy( position )
            object.position.multiplyScalar( 75 )
            object.scale.multiplyScalar( 25 )
            root.add( object )
            var atom = json.atoms[ i ]
            var text = document.createElement( 'div' )
        }
        positions = geometryBonds.getAttribute( 'position' )
        var start = new THREE.Vector3()
        var end = new THREE.Vector3()
        for ( var i = 0; i < positions.count; i += 2 ) {
            start.x = positions.getX( i )
            start.y = positions.getY( i )
            start.z = positions.getZ( i )
            end.x = positions.getX( i + 1 )
            end.y = positions.getY( i + 1 )
            end.z = positions.getZ( i + 1 )
            start.multiplyScalar( 75 )
            end.multiplyScalar( 75 )
            var object = new THREE.Mesh( boxGeometry, new THREE.MeshPhongMaterial( 0xffffff ) )
            object.position.copy( start )
            object.position.lerp( end, 0.5 )
            object.scale.set( 5, 5, start.distanceTo( end ) )
            object.lookAt( end )
            root.add( object )
        }
        render()
    } )
}
//
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize( window.innerWidth/2, window.innerHeight/2 )
    render()
}
function animate() {
    requestAnimationFrame( animate )
    controls.update()
    var time = Date.now() * 0.0004
    root.rotation.x = time * 0.5
    root.rotation.y = time * 0.5
    render()
}
function render() {
    renderer.render( scene, camera )
}