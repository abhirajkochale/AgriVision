// =====================================================================
// AGRIVISION AI
// Climate-Aware Crop Health Monitoring and Decision Support System
//
// FINAL 5-YEAR WEEKLY DATASET
//
// Study Area:
// Kavathe + Khanapur, Wai Taluka, Satara, Maharashtra
//
// Historical Period:
// 2021-01-01 to 2026-01-01
//
// Years:
// 2021, 2022, 2023, 2024, 2025
//
// Temporal Resolution:
// Weekly
//
// Variables:
// NDVI
// Rainfall_mm
// Temperature_C
// LST_C
//
// IMPORTANT:
// No interpolation.
// No artificial values.
// No fabricated satellite observations.
// =====================================================================


// =====================================================================
// 1. STUDY PERIOD
// =====================================================================

var START_DATE = ee.Date('2021-01-01');
var END_DATE = ee.Date('2026-01-01');


// =====================================================================
// 2. STUDY AREA
// =====================================================================

var kavathe = ee.Geometry.Point([
    73.9844,
    17.9467
]);

var khanapur = ee.Geometry.Point([
    73.9396,
    17.9385
]);

var aoi = kavathe
    .buffer(4000)
    .union(
        khanapur.buffer(4000)
    );

Map.centerObject(
    aoi,
    12
);

Map.addLayer(
    aoi,
    { color: 'red' },
    'Study Area',
    true
);


// =====================================================================
// 3. PROCESSING PARAMETERS
// =====================================================================

// Sentinel-2 scene-level cloud threshold.
var S2_SCENE_CLOUD_MAX = 90;

// Sentinel-2 pixel-level cloud probability threshold.
var S2_CLOUD_PROBABILITY_MAX = 70;

// Wider temporal window to improve clear observation availability.
var S2_WINDOW_BEFORE = 7;
var S2_WINDOW_AFTER = 7;

// Wider MODIS window because MOD11A2 is an 8-day product.
var MODIS_WINDOW_BEFORE = 7;
var MODIS_WINDOW_AFTER = 7;


// =====================================================================
// 4. SENTINEL-2 SURFACE REFLECTANCE
// =====================================================================

var s2 = ee.ImageCollection(
    'COPERNICUS/S2_SR_HARMONIZED'
)
    .filterBounds(
        aoi
    )
    .filterDate(
        START_DATE,
        END_DATE
    )
    .filter(
        ee.Filter.lte(
            'CLOUDY_PIXEL_PERCENTAGE',
            S2_SCENE_CLOUD_MAX
        )
    );


// =====================================================================
// 5. SENTINEL-2 CLOUD PROBABILITY
// =====================================================================

var s2CloudProbability = ee.ImageCollection(
    'COPERNICUS/S2_CLOUD_PROBABILITY'
)
    .filterBounds(
        aoi
    )
    .filterDate(
        START_DATE,
        END_DATE
    );


// =====================================================================
// 6. JOIN SENTINEL-2 WITH CLOUD PROBABILITY
// =====================================================================

var s2Joined = ee.Join.saveFirst(
    'cloud_probability'
).apply({

    primary: s2,

    secondary: s2CloudProbability,

    condition: ee.Filter.equals({

        leftField: 'system:index',

        rightField: 'system:index'

    })

});


// =====================================================================
// 7. KEEP ONLY MATCHED IMAGES
// =====================================================================

s2Joined = ee.ImageCollection(
    s2Joined
).filter(
    ee.Filter.notNull([
        'cloud_probability'
    ])
);


// =====================================================================
// 8. APPLY PIXEL-LEVEL CLOUD MASK
// =====================================================================

var s2Masked = s2Joined.map(
    function (image) {

        image = ee.Image(
            image
        );

        var cloudProbability = ee.Image(
            image.get(
                'cloud_probability'
            )
        ).select(
            'probability'
        );

        var clearPixels =
            cloudProbability.lt(
                S2_CLOUD_PROBABILITY_MAX
            );

        return image.updateMask(
            clearPixels
        );

    }
);

s2Masked = ee.ImageCollection(
    s2Masked
);


// =====================================================================
// 9. SELECT SENTINEL-2 BANDS
// =====================================================================

var s2B4B8 = s2Masked.select([
    'B4',
    'B8'
]);


// =====================================================================
// 10. CHIRPS DAILY RAINFALL
// =====================================================================

var chirps = ee.ImageCollection(
    'UCSB-CHG/CHIRPS/DAILY'
)
    .filterBounds(
        aoi
    )
    .filterDate(
        START_DATE,
        END_DATE
    )
    .select([
        'precipitation'
    ]);


// =====================================================================
// 11. ERA5-LAND DAILY TEMPERATURE
// =====================================================================

var era5 = ee.ImageCollection(
    'ECMWF/ERA5_LAND/DAILY_AGGR'
)
    .filterBounds(
        aoi
    )
    .filterDate(
        START_DATE,
        END_DATE
    )
    .select([
        'temperature_2m'
    ]);


// =====================================================================
// 12. MODIS 8-DAY LAND SURFACE TEMPERATURE
// =====================================================================

var modis = ee.ImageCollection(
    'MODIS/061/MOD11A2'
)
    .filterBounds(
        aoi
    )
    .filterDate(
        START_DATE,
        END_DATE
    )
    .select([
        'LST_Day_1km'
    ]);


// =====================================================================
// 13. FALLBACK IMAGES
// =====================================================================
//
// These are fully masked images.
//
// They do NOT create artificial values.
// They only prevent Earth Engine from failing when a time window
// contains zero images.
//
// =====================================================================


// Sentinel-2 fallback

var s2Fallback = ee.Image(
    s2.select([
        'B4',
        'B8'
    ]).first()
).updateMask(
    ee.Image.constant(0)
);


// CHIRPS fallback

var chirpsFallback = ee.Image(
    chirps.first()
).updateMask(
    ee.Image.constant(0)
);


// ERA5 fallback

var era5Fallback = ee.Image(
    era5.first()
).updateMask(
    ee.Image.constant(0)
);


// MODIS fallback

var modisFallback = ee.Image(
    modis.first()
).updateMask(
    ee.Image.constant(0)
);


// =====================================================================
// 14. CALCULATE NUMBER OF WEEKS
// =====================================================================

var totalDays =
    END_DATE.difference(
        START_DATE,
        'day'
    );

var numberOfWeeks =
    totalDays
        .divide(7)
        .ceil();

var weekIndices =
    ee.List.sequence(
        0,
        numberOfWeeks.subtract(1)
    );


// =====================================================================
// 15. CREATE WEEKLY RECORD
// =====================================================================

var createWeeklyRecord = function (index) {

    index = ee.Number(
        index
    );


    // ===================================================================
    // WEEK START
    // ===================================================================

    var weekStart =
        START_DATE.advance(
            index.multiply(7),
            'day'
        );


    // ===================================================================
    // WEEK END
    // ===================================================================

    var weekEndCandidate =
        weekStart.advance(
            7,
            'day'
        );

    var weekEnd =
        ee.Date(
            weekEndCandidate
                .millis()
                .min(
                    END_DATE.millis()
                )
        );


    // ===================================================================
    // SENTINEL-2 WIDER WINDOW
    // ===================================================================

    var s2WindowStart =
        weekStart.advance(
            -S2_WINDOW_BEFORE,
            'day'
        );

    var s2WindowEndCandidate =
        weekEnd.advance(
            S2_WINDOW_AFTER,
            'day'
        );

    var s2WindowEnd =
        ee.Date(
            s2WindowEndCandidate
                .millis()
                .min(
                    END_DATE.millis()
                )
        );


    // ===================================================================
    // SENTINEL-2 / NDVI
    // ===================================================================

    var s2Week =
        s2B4B8.filterDate(
            s2WindowStart,
            s2WindowEnd
        );

    var sentinel2Count =
        s2Week.size();

    var s2Safe =
        s2Week.merge(
            ee.ImageCollection.fromImages([
                s2Fallback
            ])
        );

    var ndviImage =
        s2Safe
            .median()
            .normalizedDifference([
                'B8',
                'B4'
            ])
            .rename(
                'NDVI'
            );


    // ===================================================================
    // CHIRPS / RAINFALL
    // ===================================================================
    //
    // Rainfall remains EXACTLY weekly.
    //
    // ===================================================================

    var chirpsWeek =
        chirps.filterDate(
            weekStart,
            weekEnd
        );

    var chirpsCount =
        chirpsWeek.size();

    var chirpsSafe =
        chirpsWeek.merge(
            ee.ImageCollection.fromImages([
                chirpsFallback
            ])
        );

    var rainfallImage =
        chirpsSafe
            .sum()
            .rename(
                'Rainfall_mm'
            );


    // ===================================================================
    // ERA5 / TEMPERATURE
    // ===================================================================
    //
    // Temperature remains EXACTLY weekly.
    //
    // ===================================================================

    var era5Week =
        era5.filterDate(
            weekStart,
            weekEnd
        );

    var era5Count =
        era5Week.size();

    var era5Safe =
        era5Week.merge(
            ee.ImageCollection.fromImages([
                era5Fallback
            ])
        );

    var temperatureImage =
        era5Safe
            .mean()
            .subtract(
                273.15
            )
            .rename(
                'Temperature_C'
            );


    // ===================================================================
    // MODIS WIDER WINDOW
    // ===================================================================

    var modisWindowStart =
        weekStart.advance(
            -MODIS_WINDOW_BEFORE,
            'day'
        );

    var modisWindowEndCandidate =
        weekEnd.advance(
            MODIS_WINDOW_AFTER,
            'day'
        );

    var modisWindowEnd =
        ee.Date(
            modisWindowEndCandidate
                .millis()
                .min(
                    END_DATE.millis()
                )
        );


    // ===================================================================
    // MODIS / LST
    // ===================================================================

    var modisWeek =
        modis.filterDate(
            modisWindowStart,
            modisWindowEnd
        );

    var modisCount =
        modisWeek.size();

    var modisSafe =
        modisWeek.merge(
            ee.ImageCollection.fromImages([
                modisFallback
            ])
        );

    var lstImage =
        modisSafe
            .mean()
            .multiply(
                0.02
            )
            .subtract(
                273.15
            )
            .rename(
                'LST_C'
            );


    // ===================================================================
    // COMBINE ALL VARIABLES
    // ===================================================================

    var combinedImage =
        ndviImage
            .addBands(
                rainfallImage
            )
            .addBands(
                temperatureImage
            )
            .addBands(
                lstImage
            );


    // ===================================================================
    // REDUCE TO STUDY AREA
    // ===================================================================

    var values =
        combinedImage.reduceRegion({

            reducer:
                ee.Reducer.mean(),

            geometry:
                aoi,

            scale:
                1000,

            bestEffort:
                true,

            maxPixels:
                1e8,

            tileScale:
                4

        });


    // ===================================================================
    // RETURN FEATURE
    // ===================================================================

    return ee.Feature(
        null,
        {

            Week_Start:
                weekStart.format(
                    'YYYY-MM-dd'
                ),

            Week_End:
                weekEnd.format(
                    'YYYY-MM-dd'
                ),

            Week_Number:
                weekStart.get(
                    'week'
                ),

            Year:
                weekStart.get(
                    'year'
                ),

            NDVI:
                values.get(
                    'NDVI'
                ),

            Rainfall_mm:
                values.get(
                    'Rainfall_mm'
                ),

            Temperature_C:
                values.get(
                    'Temperature_C'
                ),

            LST_C:
                values.get(
                    'LST_C'
                ),

            Sentinel2_Images:
                sentinel2Count,

            CHIRPS_Images:
                chirpsCount,

            ERA5_Images:
                era5Count,

            MODIS_LST_Images:
                modisCount,

            study_area:
                'Kavathe + Khanapur, Wai, Satara, Maharashtra'

        }
    );

};


// =====================================================================
// 16. GENERATE WEEKLY DATASET
// =====================================================================

var weeklyDataset =
    ee.FeatureCollection(
        weekIndices.map(
            createWeeklyRecord
        )
    );


// =====================================================================
// 17. COMPLETE DATASET
// =====================================================================

var validDataset =
    weeklyDataset.filter(
        ee.Filter.notNull([
            'NDVI',
            'Rainfall_mm',
            'Temperature_C',
            'LST_C'
        ])
    );


// =====================================================================
// 18. MAIN DATASET INFORMATION
// =====================================================================

print(
    '======================================================'
);

print(
    'AGRIVISION AI - IMPROVED 5-YEAR DATASET'
);

print(
    '======================================================'
);

print(
    'Study Area:',
    'Kavathe + Khanapur, Wai Taluka, Satara, Maharashtra'
);

print(
    'Historical Period:',
    '2021-01-01 to 2026-01-01'
);

print(
    'Number of weekly periods:',
    numberOfWeeks
);

print(
    'Sentinel-2 scenes:',
    s2.size()
);

print(
    'Sentinel-2 scenes with cloud probability:',
    s2Masked.size()
);

print(
    'CHIRPS daily images:',
    chirps.size()
);

print(
    'ERA5-Land daily images:',
    era5.size()
);

print(
    'MODIS LST images:',
    modis.size()
);

print(
    'Generated weekly records:',
    weeklyDataset.size()
);

print(
    'Valid complete weekly records:',
    validDataset.size()
);

print(
    'Dropped incomplete records:',
    weeklyDataset.size()
        .subtract(
            validDataset.size()
        )
);


// =====================================================================
// 19. MISSING DATA DIAGNOSTICS
// =====================================================================

var ndviAvailable =
    weeklyDataset.filter(
        ee.Filter.notNull([
            'NDVI'
        ])
    );

var rainfallAvailable =
    weeklyDataset.filter(
        ee.Filter.notNull([
            'Rainfall_mm'
        ])
    );

var temperatureAvailable =
    weeklyDataset.filter(
        ee.Filter.notNull([
            'Temperature_C'
        ])
    );

var lstAvailable =
    weeklyDataset.filter(
        ee.Filter.notNull([
            'LST_C'
        ])
    );


print(
    '======================================================'
);

print(
    'MISSING DATA DIAGNOSTICS'
);

print(
    '======================================================'
);

print(
    'Total weekly records:',
    weeklyDataset.size()
);

print(
    'Weeks with NDVI:',
    ndviAvailable.size()
);

print(
    'Weeks with Rainfall:',
    rainfallAvailable.size()
);

print(
    'Weeks with Temperature:',
    temperatureAvailable.size()
);

print(
    'Weeks with LST:',
    lstAvailable.size()
);

print(
    'Complete weeks:',
    validDataset.size()
);


// =====================================================================
// 20. MISSING COUNTS
// =====================================================================

print(
    '------------------------------------------------------'
);

print(
    'MISSING WEEK COUNTS'
);

print(
    '------------------------------------------------------'
);

print(
    'Missing NDVI weeks:',
    weeklyDataset.size()
        .subtract(
            ndviAvailable.size()
        )
);

print(
    'Missing Rainfall weeks:',
    weeklyDataset.size()
        .subtract(
            rainfallAvailable.size()
        )
);

print(
    'Missing Temperature weeks:',
    weeklyDataset.size()
        .subtract(
            temperatureAvailable.size()
        )
);

print(
    'Missing LST weeks:',
    weeklyDataset.size()
        .subtract(
            lstAvailable.size()
        )
);


// =====================================================================
// 21. FIRST 20 VALID RECORDS
// =====================================================================

print(
    '======================================================'
);

print(
    'FIRST 20 VALID RECORDS'
);

print(
    '======================================================'
);

print(
    validDataset.limit(
        20
    )
);


// =====================================================================
// 22. BASIC STATISTICS
// =====================================================================

var calculateStatistics =
    function (
        collection,
        fieldName,
        label
    ) {

        var statistics =
            collection.reduceColumns({

                reducer:
                    ee.Reducer.minMax()
                        .combine({

                            reducer2:
                                ee.Reducer.mean(),

                            sharedInputs:
                                true

                        }),

                selectors: [
                    fieldName
                ]

            });

        print(
            label +
            ' - Min / Max / Mean:',
            statistics
        );

    };


calculateStatistics(
    validDataset,
    'NDVI',
    'NDVI'
);

calculateStatistics(
    validDataset,
    'Rainfall_mm',
    'Rainfall (mm/week)'
);

calculateStatistics(
    validDataset,
    'Temperature_C',
    'Temperature (C)'
);

calculateStatistics(
    validDataset,
    'LST_C',
    'Land Surface Temperature (C)'
);


// =====================================================================
// 23. YEAR-WISE DATA QUALITY
// =====================================================================

var years =
    ee.List.sequence(
        2021,
        2025
    );

var yearlyQuality =
    ee.FeatureCollection(
        years.map(
            function (year) {

                year =
                    ee.Number(
                        year
                    );

                var yearDataset =
                    weeklyDataset.filter(
                        ee.Filter.eq(
                            'Year',
                            year
                        )
                    );

                var yearValid =
                    validDataset.filter(
                        ee.Filter.eq(
                            'Year',
                            year
                        )
                    );

                return ee.Feature(
                    null,
                    {

                        Year:
                            year,

                        Total_Weeks:
                            yearDataset.size(),

                        Complete_Weeks:
                            yearValid.size(),

                        Missing_Weeks:
                            yearDataset.size()
                                .subtract(
                                    yearValid.size()
                                )

                    }
                );

            }
        )
    );


print(
    '======================================================'
);

print(
    'YEARLY DATA QUALITY'
);

print(
    '======================================================'
);

print(
    yearlyQuality
);


// =====================================================================
// 24. OBSERVATION COUNTS
// =====================================================================

print(
    '======================================================'
);

print(
    'SENTINEL-2 OBSERVATION COUNTS'
);

print(
    '======================================================'
);

print(
    validDataset
        .select([
            'Week_Start',
            'Year',
            'Sentinel2_Images'
        ])
        .limit(
            50
        )
);


// =====================================================================
// 25. NDVI VISUALIZATION
// =====================================================================

var visualizationStart =
    ee.Date(
        '2025-12-01'
    );

var visualizationEnd =
    ee.Date(
        '2026-01-01'
    );

var s2Visualization =
    ee.ImageCollection(
        'COPERNICUS/S2_SR_HARMONIZED'
    )
        .filterBounds(
            aoi
        )
        .filterDate(
            visualizationStart,
            visualizationEnd
        )
        .filter(
            ee.Filter.lte(
                'CLOUDY_PIXEL_PERCENTAGE',
                S2_SCENE_CLOUD_MAX
            )
        )
        .select([
            'B4',
            'B8'
        ]);

var visualizationCount =
    s2Visualization.size();

var visualizationSafe =
    s2Visualization.merge(
        ee.ImageCollection.fromImages([
            s2Fallback
        ])
    );

var visualizationNDVI =
    visualizationSafe
        .median()
        .normalizedDifference([
            'B8',
            'B4'
        ])
        .rename(
            'NDVI'
        );

Map.addLayer(

    visualizationNDVI.clip(
        aoi
    ),

    {
        min: 0,
        max: 1,

        palette: [
            'brown',
            'yellow',
            'green'
        ]

    },

    'NDVI - December 2025',

    false

);

print(
    'Sentinel-2 images for visualization:',
    visualizationCount
);


// =====================================================================
// 26. EXPORT
// =====================================================================

var exportFields = [

    'Week_Start',
    'Week_End',
    'Week_Number',
    'Year',

    'NDVI',
    'Rainfall_mm',
    'Temperature_C',
    'LST_C',

    'Sentinel2_Images',
    'CHIRPS_Images',
    'ERA5_Images',
    'MODIS_LST_Images',

    'study_area'

];


Export.table.toDrive({

    collection:
        validDataset,

    description:
        'AgriVision_Weekly_Dataset_2021_2025_IMPROVED',

    folder:
        'AgriVision_AI',

    fileNamePrefix:
        'AgriVision_Weekly_Dataset_2021_2025_IMPROVED',

    fileFormat:
        'CSV',

    selectors:
        exportFields

});


print(
    '======================================================'
);

print(
    'EXPORT TASK CREATED'
);

print(
    'Task: AgriVision_Weekly_Dataset_2021_2025_IMPROVED'
);

print(
    'Go to the Tasks tab and click RUN ONLY AFTER CHECKING THE CONSOLE.'
);

print(
    '======================================================'
);