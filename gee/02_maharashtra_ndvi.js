// ============================================================================
// AGRIVISION AI
// SENTINEL-2 WEEKLY NDVI - 26 WEEK BATCH
//
// CURRENT BATCH:
// 2022 W27 - W52
//
// MAXIMUM POSSIBLE RECORDS:
// 167 × 26 = 4,342
//
// METHODOLOGY
// -----------
// - Sentinel-2 SR Harmonized
// - Cloud Probability < 65%
// - Scene cloudiness <= 80%
// - SCL masking
// - Weekly median NDVI
// - Exact sampling-point extraction
// - No data fabrication
//
// SPECIAL HANDLING
// ----------------
// If a week has ZERO Sentinel-2 scenes, that week contributes ZERO records.
// No empty image is passed to reduceRegions(), preventing:
// "Image has no bands."
// ============================================================================


// ============================================================================
// 1. RUN SETTINGS
// ============================================================================

var EXPORT_YEAR = 2025;

var START_WEEK = 27;
var END_WEEK = 52;


// ============================================================================
// 2. PARAMETERS
// ============================================================================

var CLOUD_PROBABILITY_THRESHOLD = 65;
var SCENE_CLOUD_THRESHOLD = 80;
var SCALE = 10;


// ============================================================================
// 3. LOAD OFFICIAL AGRIVISION SAMPLING POINTS
// ============================================================================

var originalPoints = ee.FeatureCollection(
    'projects/agrivision-505417/assets/Maharashtra_AgriVision_Sampling_Points'
);


// ============================================================================
// 4. REBUILD COORDINATES FROM ACTUAL GEOMETRY
// ============================================================================

var points = originalPoints.map(function (feature) {

    var coordinates = feature.geometry().coordinates();

    return feature.set({

        longitude: coordinates.get(0),

        latitude: coordinates.get(1)

    });

});


// ============================================================================
// 5. DATE RANGE
// ============================================================================

var yearStart = ee.Date.fromYMD(
    EXPORT_YEAR,
    1,
    1
);

var batchStart = yearStart.advance(
    START_WEEK - 1,
    'week'
);

var batchEnd = yearStart.advance(
    END_WEEK,
    'week'
);


// ============================================================================
// 6. BASIC REPORT
// ============================================================================

print(
    '=============================================='
);

print(
    'AGRIVISION - SENTINEL-2 NDVI'
);

print(
    EXPORT_YEAR +
    ' W' +
    START_WEEK +
    ' - W' +
    END_WEEK
);

print(
    '=============================================='
);

print(
    'Export year:',
    EXPORT_YEAR
);

print(
    'Start week:',
    START_WEEK
);

print(
    'End week:',
    END_WEEK
);

print(
    'Batch start:',
    batchStart.format(
        'YYYY-MM-dd'
    )
);

print(
    'Batch end:',
    batchEnd.format(
        'YYYY-MM-dd'
    )
);

print(
    'Official sampling points:',
    points.size()
);

print(
    'Maximum possible records:',
    points.size().multiply(
        END_WEEK - START_WEEK + 1
    )
);


// ============================================================================
// 7. LOAD SENTINEL-2
// ============================================================================

var sentinel2 = ee.ImageCollection(
    'COPERNICUS/S2_SR_HARMONIZED'
)
    .filterDate(
        batchStart,
        batchEnd
    )
    .filterBounds(
        points.geometry()
    )
    .filter(
        ee.Filter.lte(
            'CLOUDY_PIXEL_PERCENTAGE',
            SCENE_CLOUD_THRESHOLD
        )
    );


// ============================================================================
// 8. LOAD CLOUD PROBABILITY
// ============================================================================

var cloudProbability = ee.ImageCollection(
    'COPERNICUS/S2_CLOUD_PROBABILITY'
)
    .filterDate(
        batchStart,
        batchEnd
    )
    .filterBounds(
        points.geometry()
    );


print(
    'Sentinel-2 scenes in batch:',
    sentinel2.size()
);

print(
    'Cloud probability scenes in batch:',
    cloudProbability.size()
);


// ============================================================================
// 9. JOIN SENTINEL-2 + CLOUD PROBABILITY
// ============================================================================

var joined = ee.Join.inner().apply(

    sentinel2,

    cloudProbability,

    ee.Filter.equals({

        leftField:
            'system:index',

        rightField:
            'system:index'

    })

);


print(
    'Joined scenes in batch:',
    joined.size()
);


// ============================================================================
// 10. CREATE CLOUD-MASKED NDVI COLLECTION
// ============================================================================

var ndviCollection = ee.ImageCollection(

    joined.map(
        function (pair) {

            var image = ee.Image(
                pair.get('primary')
            );

            var cloudImage = ee.Image(
                pair.get('secondary')
            );


            // ------------------------------------------------------
            // CLOUD PROBABILITY MASK
            // ------------------------------------------------------

            var cloudMask =
                cloudImage
                    .select('probability')
                    .lt(
                        CLOUD_PROBABILITY_THRESHOLD
                    );


            // ------------------------------------------------------
            // SCL MASK
            // ------------------------------------------------------
            //
            // 3  = cloud shadow
            // 8  = medium probability cloud
            // 9  = high probability cloud
            // 10 = cirrus
            // 11 = snow / ice
            // ------------------------------------------------------

            var scl =
                image.select('SCL');


            var sclMask =
                scl
                    .neq(3)
                    .and(
                        scl.neq(8)
                    )
                    .and(
                        scl.neq(9)
                    )
                    .and(
                        scl.neq(10)
                    )
                    .and(
                        scl.neq(11)
                    );


            // ------------------------------------------------------
            // FINAL VALID MASK
            // ------------------------------------------------------

            var validMask =
                cloudMask.and(
                    sclMask
                );


            // ------------------------------------------------------
            // NDVI
            // B8 = NIR
            // B4 = RED
            // ------------------------------------------------------

            var ndvi =
                image
                    .normalizedDifference([
                        'B8',
                        'B4'
                    ])
                    .rename(
                        'NDVI'
                    )
                    .updateMask(
                        validMask
                    );


            return ndvi.copyProperties(
                image,
                [
                    'system:time_start',
                    'system:index'
                ]
            );

        }
    )

);


print(
    'Usable NDVI images in batch:',
    ndviCollection.size()
);


// ============================================================================
// 11. WEEK NUMBERS
// ============================================================================

var weekNumbers = ee.List.sequence(
    START_WEEK,
    END_WEEK
);


// ============================================================================
// 12. PROCESS WEEKS SAFELY
// ============================================================================
//
// IMPORTANT:
// We do NOT create an empty image for a week with no Sentinel-2 images.
//
// Instead:
//
// weeklyImages > 0
//     -> median -> reduceRegions
//
// weeklyImages = 0
//     -> empty FeatureCollection
//
// This prevents the "Image has no bands" error.
// ============================================================================

var emptyCollection =
    ee.FeatureCollection([]);


var finalDataset =
    ee.FeatureCollection(

        weekNumbers.iterate(

            function (
                weekNumber,
                accumulated
            ) {

                accumulated =
                    ee.FeatureCollection(
                        accumulated
                    );


                weekNumber =
                    ee.Number(
                        weekNumber
                    );


                // ----------------------------------------------------
                // CURRENT WEEK DATES
                // ----------------------------------------------------

                var weekStart =
                    yearStart.advance(
                        weekNumber.subtract(1),
                        'week'
                    );


                var weekEnd =
                    weekStart.advance(
                        1,
                        'week'
                    );


                // ----------------------------------------------------
                // CURRENT WEEK'S NDVI IMAGES
                // ----------------------------------------------------

                var weeklyImages =
                    ndviCollection.filterDate(
                        weekStart,
                        weekEnd
                    );


                // ----------------------------------------------------
                // IF IMAGES EXIST
                // ----------------------------------------------------

                var weekDataset =
                    ee.FeatureCollection(

                        ee.Algorithms.If(

                            weeklyImages.size().gt(0),

                            // ==================================================
                            // NORMAL WEEK
                            // ==================================================

                            (function () {

                                var weeklyNDVI =
                                    weeklyImages
                                        .median()
                                        .rename(
                                            'NDVI'
                                        );


                                var sampled =
                                    weeklyNDVI.reduceRegions({

                                        collection:
                                            points,

                                        reducer:
                                            ee.Reducer.mean(),

                                        scale:
                                            SCALE,

                                        tileScale:
                                            4

                                    });


                                var validSamples =
                                    sampled.filter(
                                        ee.Filter.notNull([
                                            'mean'
                                        ])
                                    );


                                var records =
                                    validSamples.map(
                                        function (feature) {

                                            return ee.Feature(
                                                null,
                                                {

                                                    // --------------------------------------
                                                    // LOCATION
                                                    // --------------------------------------

                                                    point_id:
                                                        feature.get(
                                                            'point_id'
                                                        ),

                                                    latitude:
                                                        feature.get(
                                                            'latitude'
                                                        ),

                                                    longitude:
                                                        feature.get(
                                                            'longitude'
                                                        ),

                                                    district:
                                                        feature.get(
                                                            'district'
                                                        ),

                                                    sampling_priority:
                                                        feature.get(
                                                            'sampling_priority'
                                                        ),

                                                    region_type:
                                                        feature.get(
                                                            'region_type'
                                                        ),


                                                    // --------------------------------------
                                                    // TIME
                                                    // --------------------------------------

                                                    Week_Start:
                                                        weekStart.format(
                                                            'YYYY-MM-dd'
                                                        ),

                                                    Week_End:
                                                        weekEnd.format(
                                                            'YYYY-MM-dd'
                                                        ),

                                                    Week_Number:
                                                        weekNumber,

                                                    Year:
                                                        EXPORT_YEAR,


                                                    // --------------------------------------
                                                    // NDVI
                                                    // --------------------------------------

                                                    NDVI:
                                                        feature.get(
                                                            'mean'
                                                        ),

                                                    NDVI_Valid:
                                                        1,


                                                    // --------------------------------------
                                                    // AVAILABILITY
                                                    // --------------------------------------

                                                    Sentinel2_Images:
                                                        weeklyImages.size(),


                                                    // --------------------------------------
                                                    // METHOD
                                                    // --------------------------------------

                                                    Cloud_Probability_Threshold:
                                                        CLOUD_PROBABILITY_THRESHOLD,

                                                    Scene_Cloud_Threshold:
                                                        SCENE_CLOUD_THRESHOLD,


                                                    // --------------------------------------
                                                    // DATASET
                                                    // --------------------------------------

                                                    study_area:
                                                        'Maharashtra',

                                                    dataset:
                                                        'Sentinel-2 NDVI'

                                                }
                                            );

                                        }
                                    );


                                return records;

                            })(),

                            // ==================================================
                            // ZERO-IMAGE WEEK
                            // ==================================================

                            ee.FeatureCollection([])

                        )

                    );


                // ----------------------------------------------------
                // MERGE CURRENT WEEK
                // ----------------------------------------------------

                return accumulated.merge(
                    weekDataset
                );

            },

            emptyCollection

        )

    );


// ============================================================================
// 13. FINAL COLUMN ORDER
// ============================================================================

finalDataset =
    finalDataset.select([

        'Week_Start',

        'Week_End',

        'Week_Number',

        'Year',

        'point_id',

        'latitude',

        'longitude',

        'district',

        'sampling_priority',

        'region_type',

        'NDVI',

        'NDVI_Valid',

        'Sentinel2_Images',

        'Cloud_Probability_Threshold',

        'Scene_Cloud_Threshold',

        'study_area',

        'dataset'

    ]);


// ============================================================================
// 14. FINAL REPORT
// ============================================================================

print(
    '----------------------------------------------'
);

print(
    'FINAL VALID NDVI RECORDS:',
    finalDataset.size()
);

print(
    'MAXIMUM POSSIBLE RECORDS:',
    points.size().multiply(
        END_WEEK - START_WEEK + 1
    )
);

print(
    'EXPECTED WEEK COUNT:',
    END_WEEK - START_WEEK + 1
);


// ============================================================================
// 15. SAMPLE RECORDS
// ============================================================================

print(
    'SAMPLE RECORDS:',
    finalDataset.limit(10)
);


// ============================================================================
// 16. MAP
// ============================================================================

Map.centerObject(
    points,
    6
);

Map.addLayer(
    points,
    {
        color:
            'red',

        pointSize:
            3
    },
    'AgriVision Sampling Points'
);


// ============================================================================
// 17. EXPORT
// ============================================================================

Export.table.toDrive({

    collection:
        finalDataset,

    description:
        'AgriVision_Maharashtra_NDVI_' +
        EXPORT_YEAR +
        '_W' +
        START_WEEK +
        '_W' +
        END_WEEK,

    folder:
        'AgriVision',

    fileFormat:
        'CSV',

    selectors: [

        'Week_Start',

        'Week_End',

        'Week_Number',

        'Year',

        'point_id',

        'latitude',

        'longitude',

        'district',

        'sampling_priority',

        'region_type',

        'NDVI',

        'NDVI_Valid',

        'Sentinel2_Images',

        'Cloud_Probability_Threshold',

        'Scene_Cloud_Threshold',

        'study_area',

        'dataset'

    ]

});


// ============================================================================
// END OF SCRIPT
// ============================================================================