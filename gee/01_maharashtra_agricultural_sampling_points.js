// ==============================================================================
// PHASE 1A: MAHARASHTRA AGRICULTURAL SAMPLING POINTS
// Project: AgriVision AI
// Description: Generates ~100 agricultural sampling points across Maharashtra.
// Priority is given to Satara and Latur districts.
// A specific case study point is guaranteed near Kavathe + Khanapur.
// Filtering uses ESA WorldCover 2021 (Cropland Class 40).
// ==============================================================================

// 1. Define Regions
var maharashtra = ee.FeatureCollection("FAO/GAUL/2015/level1")
  .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'));

var districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'));

// 2. Agricultural Land Mask (ESA WorldCover)
var worldcover = ee.ImageCollection("ESA/WorldCover/v200").first();
var cropland = worldcover.eq(40);

// Calculate cropland percentage in a ~500m radius using a smoothing kernel
// This evaluates the surrounding area rather than a single pixel.
var croplandKernel = ee.Kernel.circle(500, 'meters');
var croplandMean = cropland.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: croplandKernel
}).rename('cropland_percentage');

var CROPLAND_THRESHOLD = 0.50; // At least 50% cropland in the 500m buffer

// 3. Generate Regional Points
var generateDistrictPoints = function (district) {
  var dName = district.getString('ADM2_NAME');

  // Determine priority
  var isSatara = ee.String(dName).compareTo('Satara').eq(0);
  var isLatur = ee.String(dName).compareTo('Latur').eq(0);
  var isPriority = isSatara.or(isLatur);

  // Set sampling targets (15 for priority, 2 for others)
  var targetPts = ee.Algorithms.If(isPriority, 15, 2);
  var candPts = ee.Algorithms.If(isPriority, 500, 100);
  var priorityLabel = ee.Algorithms.If(isPriority, 'high', 'normal');

  // Generate candidate points
  var candidates = ee.FeatureCollection.randomPoints(district.geometry(), candPts, 42);

  // Sample the cropland percentage
  var sampled = croplandMean.sampleRegions({
    collection: candidates,
    scale: 10,
    geometries: true
  });

  // Filter candidates by agricultural threshold
  var qualified = sampled.filter(ee.Filter.gte('cropland_percentage', CROPLAND_THRESHOLD));

  // Randomly select the target number of points (implicitly ensures spatial spread)
  return qualified.randomColumn('random_sort', 42)
    .sort('random_sort')
    .limit(targetPts)
    .map(function (f) {
      return f.set({
        'district': dName,
        'sampling_priority': priorityLabel
      });
    });
};

var regionalPoints = ee.FeatureCollection(districts.map(generateDistrictPoints)).flatten();

// 4. Generate Case Study Point (Kavathe + Khanapur, Satara)
// Define approximate centroid of Kavathe/Khanapur
var caseStudyGeom = ee.Geometry.Point([73.958, 17.941]);

// Search within a 2km radius to find a highly agricultural point
var localCandidates = ee.FeatureCollection.randomPoints(caseStudyGeom.buffer(2000), 100, 42);
var localSampled = croplandMean.sampleRegions({
  collection: localCandidates,
  scale: 10,
  geometries: true
});

// Take the most agricultural point near the case study area
var caseStudyPt = localSampled.filter(ee.Filter.gte('cropland_percentage', CROPLAND_THRESHOLD))
  .sort('cropland_percentage', false)
  .first();

caseStudyPt = ee.Feature(caseStudyPt).set({
  'district': 'Satara',
  'sampling_priority': 'case_study'
});

var caseStudyFc = ee.FeatureCollection([caseStudyPt]);

// 5. Combine and Assign Point IDs
var allPoints = regionalPoints.merge(caseStudyFc);

// Convert to list to assign sequential IDs
var pointsList = allPoints.toList(500);

var finalPoints = ee.FeatureCollection(pointsList.map(function (f) {
  var feat = ee.Feature(f);
  // Get index and create ID: P001, P002, etc.
  var idx = pointsList.indexOf(f).add(1);
  var idStr = ee.String('P').cat(idx.format('%03d'));

  var coords = feat.geometry().coordinates();
  return feat.set({
    'point_id': idStr,
    'longitude': coords.get(0),
    'latitude': coords.get(1)
  });
}));

// 6. Print Statistics & Validation Information
print('=== AGRIVISION SAMPLING POINTS REPORT ===');
print('Total Final Points:', finalPoints.size());

var sataraPoints = finalPoints.filter(ee.Filter.eq('district', 'Satara'));
print('Satara Points Count:', sataraPoints.size());

var laturPoints = finalPoints.filter(ee.Filter.eq('district', 'Latur'));
print('Latur Points Count:', laturPoints.size());

var caseStudyValid = finalPoints.filter(ee.Filter.eq('sampling_priority', 'case_study'));
print('Case Study Point (Kavathe/Khanapur):', caseStudyValid);

print('Cropland Threshold Used:', CROPLAND_THRESHOLD);
print('Final FeatureCollection:', finalPoints);

// 7. Visualizations
Map.centerObject(maharashtra, 6);
Map.addLayer(maharashtra, { color: 'black' }, 'Maharashtra Boundary', false);

// Show cropland mask for context
Map.addLayer(cropland.selfMask(), { palette: ['green'] }, 'ESA Cropland (Class 40)', false);

// Show all final points
Map.addLayer(finalPoints, { color: 'blue' }, 'Final Sampling Points');

// Highlight priority regions and case study
Map.addLayer(sataraPoints, { color: 'orange' }, 'Satara Points');
Map.addLayer(laturPoints, { color: 'red' }, 'Latur Points');
Map.addLayer(caseStudyValid, { color: 'yellow' }, 'Case Study Point');

// 8. Export as CSV
Export.table.toDrive({
  collection: finalPoints,
  description: 'Maharashtra_AgriVision_Sampling_Points',
  folder: 'AgriVision',
  fileFormat: 'CSV',
  selectors: ['point_id', 'latitude', 'longitude', 'cropland_percentage', 'district', 'sampling_priority']
});
