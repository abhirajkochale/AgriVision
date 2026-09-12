// ============================================================
// AGRIVISION AI
// MAHARASHTRA WEEKLY TEMPERATURE DATASET
// 2025 - W27 TO W52
// ============================================================


// ------------------------------------------------------------
// 1. SAMPLING POINTS
// ------------------------------------------------------------

var points = ee.FeatureCollection(
    'projects/agrivision-505417/assets/Maharashtra_AgriVision_Sampling_Points'
);

print('Sampling points:', points.size());


// ------------------------------------------------------------
// 2. STUDY AREA
// ------------------------------------------------------------

var studyArea = points.geometry().bounds();


// ------------------------------------------------------------
// 3. ERA5-LAND DAILY AGGREGATED
// ------------------------------------------------------------

var era5 = ee.ImageCollection(
    'ECMWF/ERA5_LAND/DAILY_AGGR'
)
    .filterDate(
        '2025-06-25',
        '2026-01-01'
    )
    .filterBounds(studyArea)
    .select('temperature_2m');

print('ERA5-Land images:', era5.size());


// ------------------------------------------------------------
// 4. FIXED START DATE
// ------------------------------------------------------------

var startDate = ee.Date('2025-01-01');


// ------------------------------------------------------------
// 5. WEEK NUMBERS
// ------------------------------------------------------------

var weekNumbers = ee.List.sequence(27, 52);


// ------------------------------------------------------------
// 6. CREATE WEEKLY DATA
// ------------------------------------------------------------

var weeklyCollections = weekNumbers.map(function (weekNumber) {

    weekNumber = ee.Number(weekNumber);


    // ----------------------------------------------------------
    // WEEK START / END
    // ----------------------------------------------------------

    var weekStart = startDate.advance(
        weekNumber.subtract(1).multiply(7),
        'day'
    );

    var weekEnd = weekStart.advance(
        7,
        'day'
    );


    // ----------------------------------------------------------
    // DAILY ERA5 DATA FOR THIS WEEK
    // ----------------------------------------------------------

    var weekData = era5.filterDate(
        weekStart,
        weekEnd
    );

    var dayCount = weekData.size();


    // ----------------------------------------------------------
    // WEEKLY MEAN TEMPERATURE
    // ----------------------------------------------------------

    var weeklyMeanK = weekData.mean();

    var weeklyMeanC = weeklyMeanK
        .subtract(273.15)
        .rename('Temperature_C');


    // ----------------------------------------------------------
    // SAMPLE 167 FIXED POINTS
    // ----------------------------------------------------------

    var reduced = weeklyMeanC.reduceRegions({

        collection: points,

        reducer: ee.Reducer.mean(),

        scale: 11132,

        tileScale: 4

    });


    // ----------------------------------------------------------
    // ADD METADATA
    // ----------------------------------------------------------

    return reduced.map(function (feature) {

        var coords = feature.geometry().coordinates();

        var temperature = feature.get('mean');

        return ee.Feature(null, {

            Week_Start:
                weekStart.format('YYYY-MM-dd'),

            Week_End:
                weekEnd.format('YYYY-MM-dd'),

            Week_Number:
                weekNumber,

            Year:
                2025,

            point_id:
                feature.get('point_id'),

            latitude:
                coords.get(1),

            longitude:
                coords.get(0),

            district:
                feature.get('district'),

            sampling_priority:
                feature.get('sampling_priority'),

            region_type:
                feature.get('region_type'),

            Temperature_C:
                temperature,

            Temperature_Valid:
                ee.Algorithms.If(
                    temperature,
                    1,
                    0
                ),

            ERA5_Days:
                dayCount,

            temperature_source:
                'ERA5-Land Daily Aggregated',

            study_area:
                'Maharashtra',

            dataset:
                'ERA5-Land Temperature'

        });

    });

});


// ------------------------------------------------------------
// 7. FLATTEN
// ------------------------------------------------------------

var temperatureWeekly = ee.FeatureCollection(
    weeklyCollections
).flatten();


// ------------------------------------------------------------
// 8. FINAL COLUMNS
// ------------------------------------------------------------

temperatureWeekly = temperatureWeekly.select([

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
    'Temperature_C',
    'Temperature_Valid',
    'ERA5_Days',
    'temperature_source',
    'study_area',
    'dataset'

]);


// ------------------------------------------------------------
// 9. VALIDATION
// ------------------------------------------------------------

print('--------------------------------------------');
print('TEMPERATURE DATASET - 2025 W27-W52');
print('--------------------------------------------');

print(
    'Expected records:',
    167 * 26
);

print(
    'Total records:',
    temperatureWeekly.size()
);

print(
    'Unique point IDs:',
    temperatureWeekly
        .aggregate_count_distinct('point_id')
);

print(
    'Unique weeks:',
    temperatureWeekly
        .aggregate_count_distinct('Week_Start')
);

print(
    'Minimum temperature (C):',
    temperatureWeekly
        .aggregate_min('Temperature_C')
);

print(
    'Maximum temperature (C):',
    temperatureWeekly
        .aggregate_max('Temperature_C')
);

print(
    'Mean temperature (C):',
    temperatureWeekly
        .aggregate_mean('Temperature_C')
);

print(
    'Minimum ERA5 days:',
    temperatureWeekly
        .aggregate_min('ERA5_Days')
);

print(
    'Maximum ERA5 days:',
    temperatureWeekly
        .aggregate_max('ERA5_Days')
);

print(
    'Valid temperature records:',
    temperatureWeekly
        .filter(
            ee.Filter.eq(
                'Temperature_Valid',
                1
            )
        )
        .size()
);

print(
    'Sample records:',
    temperatureWeekly.limit(10)
);


// ------------------------------------------------------------
// 10. MAP
// ------------------------------------------------------------

Map.centerObject(points, 6);

Map.addLayer(
    points,
    { color: 'red' },
    'AgriVision Sampling Points'
);


// ------------------------------------------------------------
// 11. EXPORT
// ------------------------------------------------------------

Export.table.toDrive({

    collection:
        temperatureWeekly,

    description:
        'AgriVision_Maharashtra_Temperature_2025_W27_W52',

    fileNamePrefix:
        'AgriVision_Maharashtra_Temperature_2025_W27_W52',

    folder:
        'AgriVision_Temperature',

    fileFormat:
        'CSV'

});