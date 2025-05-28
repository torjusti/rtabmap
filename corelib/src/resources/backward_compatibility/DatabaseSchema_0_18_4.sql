-- *******************************************************************
--  DatabaseSchema: Script for creating the database
--   Usage:
--       $ sqlite3 LTM.db < DatabaseSchema.sql
--
-- *******************************************************************

-- *******************************************************************
-- CLEAN
-- *******************************************************************
/*DROP TABLE Node;*/

-- *******************************************************************
-- CREATE
-- *******************************************************************
CREATE TABLE Node (
    id INTEGER NOT NULL,
    map_id INTEGER NOT NULL,
    weight INTEGER,
    stamp FLOAT,
    pose BLOB,                -- 3x4 float
    ground_truth_pose BLOB,   -- 3x4 float
    velocity BLOB,            -- 6 float (vx,vy,vz,vroll,vpitch,vyaw) m/s and rad/s
    label TEXT,
    gps BLOB,                 -- 1x6 double: stamp, longitude (DD), latitude (DD), altitude (m), accuracy (m), bearing (North 0->360 deg clockwise)
    env_sensors BLOB,         -- Variable 3xdouble: (sensorId1, value, stamp, sensorId2, value, stamp, ...)
    time_enter DATE,
    PRIMARY KEY (id)
);

CREATE TABLE Data (
    id INTEGER NOT NULL,
    image BLOB,               -- compressed image (Grayscale or RGB)
    depth BLOB,               -- compressed image (High confidence depth map)
    depth_medium BLOB,        -- compressed image (Medium confidence depth map)
    depth_low BLOB,          -- compressed image (Low confidence depth map) 
    depth_confidence BLOB,    -- compressed image (Confidence map ARKit)
    calibration BLOB,         -- fx, fy, cx, cy, [baseline,] width, height, local_transform
    
    scan BLOB,                -- compressed data (Laser scan)
    scan_info BLOB,           -- scan_max_pts, scan_max_range, scan_format, local_transform
    
    ground_cells BLOB,        -- compressed data (occupancy grid)
    obstacle_cells BLOB,      -- compressed data (occupancy grid)
    empty_cells BLOB,         -- compressed data (occupancy grid)
    cell_size FLOAT,
    view_point_x FLOAT,
    view_point_y FLOAT,
    view_point_z FLOAT,
    
    user_data BLOB,           -- compressed data (User data)
    time_enter DATE,
    PRIMARY KEY (id)
);

CREATE TABLE Link (
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    type INTEGER NOT NULL,    -- neighbor=0, loop=1, child=2
    information_matrix BLOB NOT NULL, -- 6x6 double (inverse covariance)
    transform BLOB,           -- 3x4 float
    user_data BLOB,           -- compressed data (User data)
    FOREIGN KEY (from_id) REFERENCES Node(id),
    FOREIGN KEY (to_id) REFERENCES Node(id)
);

-- 
CREATE TABLE Word (
    id INTEGER NOT NULL,
    descriptor_size INTEGER NOT NULL,
    descriptor BLOB NOT NULL,
    time_enter DATE,
    PRIMARY KEY (id)
);

CREATE TABLE Feature (
    node_id INTEGER NOT NULL,
    word_id INTEGER NOT NULL,
    pos_x FLOAT NOT NULL,
    pos_y FLOAT NOT NULL,
    size INTEGER NOT NULL,
    dir FLOAT NOT NULL,
    response FLOAT NOT NULL,
    octave INTEGER NOT NULL,
    depth_x FLOAT,
    depth_y FLOAT,
    depth_z FLOAT,
    descriptor_size INTEGER,
    descriptor BLOB,
    FOREIGN KEY (node_id) REFERENCES Node(id)
);

--
CREATE TABLE Tag (
    node_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    stamp FLOAT NOT NULL,
    transform BLOB NOT NULL,     -- 3x4 float, /base_link -> /tag_frame
    FOREIGN KEY (node_id) REFERENCES Node(id)
);

CREATE TABLE Info (
    STM_size INTEGER,
    last_sign_added INTEGER,
    optimized_poses BLOB,         -- serialized poses (compressed)
    optimizer_error FLOAT,
    working_dir TEXT,
    database_version TEXT,
    dictionary_version INTEGER,
    extended_configuration TEXT,  -- parameters
    GPS_stamp FLOAT,             -- ref GPS stamp
    GPS_cached BLOB              -- serialized GPS cache (compressed)
);

--
CREATE TABLE Statistics (
    id INTEGER NOT NULL,
    stamp FLOAT NOT NULL,
    data BLOB NOT NULL,       -- compressed data
    wm_state BLOB,           -- serialized working memory state
    FOREIGN KEY (id) REFERENCES Node(id)
);

CREATE TABLE WM_State (
    file_id INTEGER NOT NULL,
    state BLOB NOT NULL,      -- serialized list (compressed)
    FOREIGN KEY (file_id) REFERENCES Node(id)
);
