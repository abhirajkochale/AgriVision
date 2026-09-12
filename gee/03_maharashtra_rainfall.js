// ============================================================================
// AGRIVISION AI
// CHIRPS WEEKLY RAINFALL
// 26-WEEK TEST BATCH
//
// CURRENT TEST:
// 2021 W01 - W26
//
// MAXIMUM POSSIBLE RECORDS:
// 167 points × 26 weeks = 4,342
//
// SOURCE:
// CHIRPS Daily v2.0 Final
//
// DATASET:
// UCSB-CHG/CHIRPS/DAILY
//
// BAND:
// precipitation (mm/day)
//
// WEEKLY VALUE:
// Sum of 7 daily precipitation values = mm/week
//
// NO DATA FABRICATION
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

var SCALE = 5566;


// ============================================================================
// 3. LOAD OFFICIAL AGRIVISION SAMPLING POINTS
// ============================================================================

var originalPoints = ee.FeatureCollection(
    'projects/agrivision-505417/assets/Maharashtra_AgriVision_Sampling_Points'
);


// ============================================================================
// 4. REBUILD COORDINATES FROM GEOMETRY
// ============================================================================

var points = originalPoints.map(function (feature) {

    var coordinates =
        feature.geometry().coordinates();

    return feature.set({

        longitude:
            coordinates.get(0),

        latitude:
            coordinates.get(1)

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
    'AGRIVISION - CHIRPS WEEKLY RAINFALL'
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
// 7. LOAD CHIRPS DAILY
// ============================================================================

var chirps = ee.ImageCollection(
    'UCSB-CHG/CHIRPS/DAILY'
)
    .filterDate(
        batchStart,
        batchEnd
    )
    .filterBounds(
        points.geometry()
    )
    .select(
        'precipitation'
    );


print(
    'CHIRPS daily images in batch:',
    chirps.size()
);


// ============================================================================
// 8. PROCESS EACH WEEK
// ============================================================================

var weekNumbers = ee.List.sequence(
    START_WEEK,
    END_WEEK
);


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
                // CURRENT WEEK'S CHIRPS DATA
                // ----------------------------------------------------

                var weeklyImages =
                    chirps.filterDate(
                        weekStart,
                        weekEnd
                    );


                // ----------------------------------------------------
                // WEEKLY RAINFALL SUM
                // ----------------------------------------------------

                var weekDataset =
                    ee.FeatureCollection(

                        ee.Algorithms.If(

                            weeklyImages.size().gt(0),

                            // ==================================================
                            // NORMAL WEEK
                            // ==================================================

                            (function () {

                                var weeklyRainfall =
                                    weeklyImages
                                        .sum()
                                        .rename(
                                            'Rainfall_mm'
                                        );


                                // ------------------------------------------------
                                // EXTRACT AT OFFICIAL POINTS
                                // ------------------------------------------------

                                var sampled =
                                    weeklyRainfall.reduceRegions({

                                        collection:
                                            points,

                                        reducer:
                                            ee.Reducer.mean(),

                                        scale:
                                            SCALE,

                                        tileScale:
                                            4

                                    });


                                // ------------------------------------------------
                                // KEEP VALID RECORDS
                                // ------------------------------------------------

                                var validSamples =
                                    sampled.filter(
                                        ee.Filter.notNull([
                                            'mean'
                                        ])
                                    );


                                // ------------------------------------------------
                                // BUILD FINAL RECORDS
                                // ------------------------------------------------

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
                                                    // RAINFALL
                                                    // --------------------------------------

                                                    Rainfall_mm:
                                                        feature.get(
                                                            'mean'
                                                        ),

                                                    Rainfall_Valid:
                                                        1,


                                                    // --------------------------------------
                                                    // DATA AVAILABILITY
                                                    // --------------------------------------

                                                    CHIRPS_Days:
                                                        weeklyImages.size(),


                                                    // --------------------------------------
                                                    // SOURCE
                                                    // --------------------------------------

                                                    rainfall_source:
                                                        'CHIRPS Daily v2.0',

                                                    study_area:
                                                        'Maharashtra',

                                                    dataset:
                                                        'CHIRPS Rainfall'

                                                }
                                            );

                                        }
                                    );


                                return records;

                            })(),

                            // ==================================================
                            // ZERO-DATA WEEK
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
// 9. FINAL COLUMN ORDER
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

        'Rainfall_mm',

        'Rainfall_Valid',

        'CHIRPS_Days',

        'rainfall_source',

        'study_area',

        'dataset'

    ]);


// ============================================================================
// 10. FINAL REPORT
// ============================================================================

print(
    '----------------------------------------------'
);

print(
    'FINAL VALID RAINFALL RECORDS:',
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
// 11. SAMPLE RECORDS
// ============================================================================

print(
    'SAMPLE RECORDS:',
    finalDataset.limit(10)
);


// ============================================================================
// 12. MAP
// ============================================================================

Map.centerObject(
    points,
    6
);

Map.addLayer(
    points,
    {
        color:
            'blue',
        pointSize:
            3
    },
    'AgriVision Sampling Points'
);


// ============================================================================
// 13. EXPORT
// ============================================================================

Export.table.toDrive({

    collection:
        finalDataset,

    description:
        'AgriVision_Maharashtra_Rainfall_' +
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

        'Rainfall_mm',

        'Rainfall_Valid',

        'CHIRPS_Days',

        'rainfall_source',

        'study_area',

        'dataset'

    ]

});


// ============================================================================
// END OF SCRIPT
// ============================================================================