#include "global_map.h"
#include "hungarian-algorithm-cpp/Hungarian.h"

Eigen::Vector3d point_to_vec(geometry_msgs::Point point)
{
    Eigen::Vector3d vec;
    vec<<point.x, point.y, point.z;
    return vec;
}

geometry_msgs::Point vec_to_point(Eigen::Vector3d vec)
{
    geometry_msgs::Point point{};
    point.x = vec[0];
    point.y = vec[1];
    point.z = vec[2];
    return point;
}

tauv_msgs::FeatureDetection makeRosDetection(shared_ptr<Tracker> det)
 {
    tauv_msgs::FeatureDetection returnDetection{};
    returnDetection.position = vec_to_point(det->getPosition());
    returnDetection.orientation = vec_to_point(det->getOrientation());
    returnDetection.tag = det->getTag();
    returnDetection.confidence = det->getConfidence();
    returnDetection.SE2 = det->is_SE2();
    return returnDetection;
 }

GlobalMap::GlobalMap(ros::NodeHandle& handler)
{
    featureCount = 0;
    totalDetections = 0;
    DUMMY_FILL=0;
    MAP = {};

    listener = handler.subscribe("global_map/feature_detections", 100, &GlobalMap::updateTrackersInterface, this);
    resetService = handler.advertiseService("global_map/reset", &GlobalMap::reset, this);
    findService = handler.advertiseService("global_map/find", &GlobalMap::find, this);
    findOneService = handler.advertiseService("global_map/find_one", &GlobalMap::findOne, this);
    findClosestService = handler.advertiseService("global_map/find_closest", &GlobalMap::findClosest, this);
    syncDetectionsService = handler.advertiseService("global_map/sync_detections", &GlobalMap::syncAddDetections, this);
    nodeHandler = handler;
}

vector<Detection> GlobalMap::convertToStruct(vector<tauv_msgs::FeatureDetection> &detections, string tracker_type)
{
    vector<Detection> trueDets(detections.size());
    
    for(size_t i=0;i<detections.size();i++)
    {
        tauv_msgs::FeatureDetection detection = detections[i];
        trueDets[i] = Detection{point_to_vec(detection.position), point_to_vec(detection.orientation), detection.tag, tracker_type, detection.confidence, (bool)detection.SE2};
    }

    return trueDets;
}

//for interfacing w publisher
void GlobalMap::updateTrackersInterface(const tauv_msgs::FeatureDetections::ConstPtr& detectionObjects)
{
    //convert trackers into readable cpp format to save later repeated computation
    vector<tauv_msgs::FeatureDetection> objdets = detectionObjects->detections;
    updateTrackers(objdets, detectionObjects->detector_tag);
}

bool GlobalMap::updateTrackers(vector<tauv_msgs::FeatureDetection> objdets, string detector_type)
{
    vector<Detection> detections = convertToStruct(objdets, detector_type);

    size_t len = detections.size();
    
    if(len==0){
        return false;
    }

    totalDetections+=len;

    mtx.lock();

    assignDetections(detections);

    mtx.unlock();

    return true;
}

vector<pair<shared_ptr<TrackerMaster>, int>> GlobalMap::generateSimilarityMatrix(vector<Detection> &detections, vector<vector<double>> &costMatrix, size_t costMatSize)
{
    //list for detection addition
    vector<pair<shared_ptr<TrackerMaster>, int>> trackerList(featureCount);

    size_t featureNum = 0;
    for(pair<string,shared_ptr<TrackerMaster>> Tracker: MAP)
    {
        featureNum = Tracker.second->generateSimilarityMatrix(detections, costMatrix, featureNum, trackerList);
    }

    //fill with dummy trackers, only happens when new object detected
    for(size_t dummies = featureNum; dummies<costMatSize ; dummies++)
    {
        vector<double> DUMMY(detections.size());
        costMatrix[dummies] = DUMMY;
        for(size_t j=0; j<detections.size(); j++)
        {
            costMatrix[dummies][j] = DUMMY_FILL;
        }
    }

    return trackerList;
}

void GlobalMap::assignDetections(vector<Detection> &detections)
{
    size_t initTrackerNum = featureCount; //will change during iterations if new trackers added

    size_t costMatSize = max(featureCount,detections.size());
    vector<vector<double>> costMatrix(costMatSize);
    vector<pair<shared_ptr<TrackerMaster>, int>> trackerList = generateSimilarityMatrix(detections, costMatrix, costMatSize);

    //find best assignment
    vector<int> assignment;
    HungarianAlgorithm Matcher;
    Matcher.Solve(costMatrix, assignment);

    //if valid match, assign detection to tracker,
    //otherwise, create a new tracker
    for(size_t trackerIdx=0; trackerIdx<initTrackerNum;trackerIdx++)
    {
        int detectionIdx = assignment[trackerIdx];
        pair<shared_ptr<TrackerMaster>, int> master = trackerList[trackerIdx];

        //tracker had no corresponding detections
        if(detectionIdx<0){updateDecay(master.first, master.second); continue;}

        if(master.first->validCost(master.second, costMatrix[trackerIdx][detectionIdx]))
        {
            assignDetection(master.first, master.second, detections[detectionIdx]);
        }
        else 
        {
            updateDecay(master.first, master.second);
            addTracker(detections[detectionIdx]);
        }
    }

    //make new trackers for unmatched detections
    for(size_t unmatched=initTrackerNum; unmatched<costMatSize; unmatched++)
    {
        int detectionIdx = assignment[unmatched];
        addTracker(detections[detectionIdx]);
    }
}

void GlobalMap::updateDecay(shared_ptr<TrackerMaster> F, size_t featureIdx)
{
    if(F->decay(featureIdx, totalDetections))
    {
        F->deleteTracker(featureIdx);
        featureCount-=1;
    }
}

shared_ptr<TrackerMaster> GlobalMap::findTrackerMaster(string tag)
{
    unordered_map<string,shared_ptr<TrackerMaster>>::iterator NEW = MAP.find(tag);

    if(NEW == MAP.end())
    {
        shared_ptr<TrackerMaster> F (new TrackerMaster(tag, nodeHandler));
        MAP.insert({tag, F});
        DUMMY_FILL = max(DUMMY_FILL, F->getMaxThreshold());

        return F;
    }

    return NEW->second;
}

void GlobalMap::assignDetection(shared_ptr<TrackerMaster> F, size_t featureIdx, Detection &detection)
{
    // cout<<F->getTracker(featureIdx)->getTag()<<"\n";
    // cout<<detection.tag<<"\n";
    if(F->reassignable(featureIdx, detection))
    {
        shared_ptr<Tracker> NEW = F->popReassignable(featureIdx, detection);
        shared_ptr<TrackerMaster> tracker = findTrackerMaster(detection.tag);
        tracker->addTracker(NEW);
    }
    else
    {
        F->addDetection(featureIdx, detection);
    }
}

void GlobalMap::addTracker(Detection &detection)
{
    shared_ptr<TrackerMaster> tracker = findTrackerMaster(detection.tag);
    tracker->addTracker(detection);
    featureCount+=1;
}


bool GlobalMap::syncAddDetections(tauv_msgs::FeatureDetectionsSync::Request &req, tauv_msgs::FeatureDetectionsSync::Response &res)
{
    vector<tauv_msgs::FeatureDetection> objdets = req.detections.detections;
    res.success = updateTrackers(objdets, req.detections.detector_tag);
    return true;
}

bool GlobalMap::find(tauv_msgs::MapFind::Request &req, tauv_msgs::MapFind::Response &res)
{
    unordered_map<string,shared_ptr<TrackerMaster>>::iterator master = MAP.find(req.tag);

    if(master == MAP.end()){res.success=false; res.detections = {}; return true;}

    vector<shared_ptr<Tracker>> detections = (master->second)->getTrackers();

    vector<tauv_msgs::FeatureDetection> returnDetections((master->second)->getNumTrackers());

    size_t count=0;
    for(shared_ptr<Tracker> detection : detections)
    {
        if(detection->State==TrackerState::ZOMBIE){continue;}

        returnDetections[count] = makeRosDetection(detection);
        count++;
    }

    res.detections = returnDetections;
    res.success = true;

    return true;
}

bool GlobalMap::findOne(tauv_msgs::MapFindOne::Request &req, tauv_msgs::MapFindOne::Response &res)
{
    unordered_map<string,shared_ptr<TrackerMaster>>::iterator master = MAP.find(req.tag);

    if(master == MAP.end()){res.success=false; res.detection = {}; return true;}

    vector<shared_ptr<Tracker>> detections = (master->second)->getTrackers();

    if(detections.size()==0)
    {
        res.detection = {};
        res.success = false;

        return true;
    }

    int minInd = 0;
    double maxDecay = -1;
    for(size_t i = 0; i<detections.size(); i++)
    {
        if(detections[i]->State==TrackerState::ZOMBIE){continue;}

        double decay = (master->second)->getDecay(i, totalDetections);

        if(decay>maxDecay)
        {
            minInd = i;
            maxDecay = decay;
        }
    }

    res.detection = makeRosDetection(detections[minInd]);
    res.success = true;

    return true;
}

bool GlobalMap::findClosest(tauv_msgs::MapFindClosest::Request &req, tauv_msgs::MapFindClosest::Response &res)
{
    unordered_map<string,shared_ptr<TrackerMaster>>::iterator master = MAP.find(req.tag);
    Eigen::Vector3d position = point_to_vec(req.point);

    if(master == MAP.end()){res.success=false; res.detection = {}; return true;}

    vector<shared_ptr<Tracker>> detections = (master->second)->getTrackers();

    if(detections.size()==0)
    {
        res.detection = {};
        res.success = false;

        return true;
    }

    int minInd = 0;
    double minDist = -1;
    for(size_t i = 0; i<detections.size(); i++)
    {
        shared_ptr<Tracker> detection = detections[i];
        if(detection->State==TrackerState::ZOMBIE){continue;}

        double dist = (detection->getPosition() - position).norm();

        if(minDist<0 || dist<minDist)
        {
            minInd = i;
            minDist = dist;
        }
    }

    res.detection = makeRosDetection(detections[minInd]);
    res.success = true;

    return true;
}

bool GlobalMap::reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
    featureCount = 0;
    totalDetections = 0;
    DUMMY_FILL=0;
    MAP = {};

    res.success = true;
    return true;
}