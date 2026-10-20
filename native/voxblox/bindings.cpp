#include <cmath>
#include <memory>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/integrator/esdf_integrator.h>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Bind upstream TSDF/ESDF integrators without reimplementing their algorithms.
class TsdfVolume {
 public:
  TsdfVolume(double resolution, int block_side, double truncation, double max_weight,
             double max_range, int threads) {
    if (!std::isfinite(resolution) || resolution <= 0 || block_side < 2 || block_side > 64 ||
        (block_side & (block_side - 1)) || !std::isfinite(truncation) || truncation < resolution ||
        !std::isfinite(max_weight) || max_weight < 1 || !std::isfinite(max_range) ||
        max_range <= truncation || threads < 1 || threads > 32)
      throw std::invalid_argument("Invalid native Voxblox settings");
    layer_ = std::make_unique<voxblox::Layer<voxblox::TsdfVoxel>>(resolution, block_side);
    voxblox::TsdfIntegratorBase::Config config;
    config.default_truncation_distance = truncation;
    config.max_weight = max_weight;
    config.min_ray_length_m = .01;
    config.max_ray_length_m = max_range;
    config.use_const_weight = true;
    config.integrator_threads = threads;
    config.enable_anti_grazing = true;
    integrator_ = std::make_unique<voxblox::MergedTsdfIntegrator>(config, layer_.get());
  }

  void integrate(Array origin, Array endpoints) {
    if (origin.ndim() != 1 || origin.shape(0) != 3 || endpoints.ndim() != 2 || endpoints.shape(1) != 3)
      throw std::invalid_argument("Voxblox requires origin (3,) and endpoints (N,3)");
    auto o = origin.unchecked<1>(); auto p = endpoints.unchecked<2>();
    for (int d = 0; d < 3; ++d) if (!std::isfinite(o(d))) throw std::invalid_argument("Nonfinite camera origin");
    voxblox::Point position(o(0), o(1), o(2));
    voxblox::Pointcloud cloud; voxblox::Colors colors;
    cloud.reserve(endpoints.shape(0)); colors.reserve(endpoints.shape(0));
    for (py::ssize_t row = 0; row < endpoints.shape(0); ++row) {
      for (int d = 0; d < 3; ++d) if (!std::isfinite(p(row,d))) throw std::invalid_argument("Nonfinite TSDF endpoint");
      cloud.emplace_back(p(row,0)-position.x(), p(row,1)-position.y(), p(row,2)-position.z());
      colors.emplace_back(80, 180, 210);
    }
    voxblox::Transformation transform(voxblox::Quaternion(1., 0., 0., 0.), position);
    py::gil_scoped_release release;
    integrator_->integratePointCloud(transform, cloud, colors);
  }

  py::array_t<double> query(Array points) const {
    if (points.ndim() != 2 || points.shape(1) != 3) throw std::invalid_argument("Query must be (N,3)");
    py::array_t<double> result({points.shape(0), py::ssize_t(2)});
    auto p = points.unchecked<2>(); auto out = result.mutable_unchecked<2>();
    py::gil_scoped_release release;
    for (py::ssize_t row = 0; row < points.shape(0); ++row) {
      for (int d = 0; d < 3; ++d) if (!std::isfinite(p(row,d))) throw std::invalid_argument("Nonfinite TSDF query");
      const auto* voxel = layer_->getVoxelPtrByCoordinates(voxblox::Point(p(row,0),p(row,1),p(row,2)));
      out(row,0) = voxel && voxel->weight > 0 ? voxel->distance : std::numeric_limits<double>::quiet_NaN();
      out(row,1) = voxel ? voxel->weight : 0.;
    }
    return result;
  }

  py::array_t<double> observed() const {
    voxblox::BlockIndexList blocks; layer_->getAllAllocatedBlocks(&blocks);
    size_t count=0;
    for (const auto& index : blocks) {
      const auto& block=layer_->getBlockByIndex(index);
      for(size_t i=0;i<block.num_voxels();++i) if(block.getVoxelByLinearIndex(i).weight>0) ++count;
    }
    py::array_t<double> result({py::ssize_t(count),py::ssize_t(5)});
    auto out=result.mutable_unchecked<2>(); size_t row=0;
    py::gil_scoped_release release;
    for (const auto& index : blocks) {
      const auto& block=layer_->getBlockByIndex(index);
      for(size_t i=0;i<block.num_voxels();++i) {
        const auto& voxel=block.getVoxelByLinearIndex(i);
        if(voxel.weight<=0) continue;
        const auto point=block.computeCoordinatesFromLinearIndex(i);
        for(int d=0;d<3;++d) out(row,d)=point(d);
        out(row,3)=voxel.distance; out(row,4)=voxel.weight; ++row;
      }
    }
    return result;
  }

  void restore(Array points, Array values) {
    if (points.ndim()!=2 || points.shape(1)!=3 || values.ndim()!=2 ||
        values.shape(0)!=points.shape(0) || values.shape(1)!=2) throw std::invalid_argument("Invalid protected TSDF cells");
    auto p=points.unchecked<2>(); auto v=values.unchecked<2>();
    for (py::ssize_t row=0; row<points.shape(0); ++row) {
      for(int d=0;d<3;++d) if(!std::isfinite(p(row,d))) throw std::invalid_argument("Nonfinite protected coordinate");
      if(!std::isfinite(v(row,1)) || v(row,1)<0 || (v(row,1)>0 && !std::isfinite(v(row,0))))
        throw std::invalid_argument("Invalid protected TSDF values");
      const voxblox::Point point(p(row,0),p(row,1),p(row,2));
      auto* voxel=layer_->getVoxelPtrByCoordinates(point);
      if (!voxel) continue;
      voxel->distance=v(row,1)>0 ? v(row,0) : 0.; voxel->weight=v(row,1);
    }
  }

  void crop(Array low, Array high, int max_blocks) {
    if (low.ndim()!=1 || high.ndim()!=1 || low.size()!=3 || high.size()!=3 || max_blocks<1)
      throw std::invalid_argument("Invalid TSDF block bounds");
    auto a=low.unchecked<1>(); auto b=high.unchecked<1>();
    for(int d=0;d<3;++d) if(!std::isfinite(a(d)) || !std::isfinite(b(d)) || b(d)<=a(d))
      throw std::invalid_argument("Invalid TSDF block bounds");
    voxblox::BlockIndexList blocks; layer_->getAllAllocatedBlocks(&blocks);
    for (const auto& index : blocks) {
      const auto& block=layer_->getBlockByIndex(index);
      bool outside=false;
      for(int d=0; d<3; ++d) outside |= block.origin()(d)>=b(d) || block.origin()(d)+block.block_size()<=a(d);
      if(outside) layer_->removeBlock(index);
    }
    if (layer_->getNumberOfAllocatedBlocks()>static_cast<size_t>(max_blocks))
      throw std::runtime_error("Voxblox native block budget exceeded");
  }
  size_t blocks() const { return layer_->getNumberOfAllocatedBlocks(); }

  void configure_esdf(double max_distance, double fixed_distance, double min_weight) {
    if (!std::isfinite(max_distance) || !std::isfinite(fixed_distance) ||
        !std::isfinite(min_weight) || fixed_distance <= 0 ||
        max_distance <= fixed_distance || min_weight <= 0)
      throw std::invalid_argument("Invalid native ESDF settings");
    esdf_integrator_.reset();
    esdf_layer_ = std::make_unique<voxblox::Layer<voxblox::EsdfVoxel>>(
        layer_->voxel_size(), layer_->voxels_per_side());
    voxblox::EsdfIntegrator::Config config;
    config.full_euclidean_distance = true;
    config.max_distance_m = max_distance;
    config.default_distance_m = max_distance;
    config.min_distance_m = fixed_distance;
    config.min_weight = min_weight;
    config.add_occupied_crust = false;
    esdf_integrator_ = std::make_unique<voxblox::EsdfIntegrator>(
        config, layer_.get(), esdf_layer_.get());
  }

  void update_esdf() {
    if (!esdf_integrator_) throw std::runtime_error("ESDF is not configured");
    py::gil_scoped_release release;
    esdf_integrator_->clear();
    // Rebuild after ALL views, uncertain-cell restoration and workspace cropping.
    esdf_integrator_->updateFromTsdfLayerBatch();
  }

  py::array_t<double> esdf_observed() const {
    if (!esdf_layer_) throw std::runtime_error("ESDF is not configured");
    voxblox::BlockIndexList blocks; esdf_layer_->getAllAllocatedBlocks(&blocks);
    size_t count=0;
    for (const auto& index : blocks) {
      const auto& block=esdf_layer_->getBlockByIndex(index);
      for(size_t i=0;i<block.num_voxels();++i)
        if(block.getVoxelByLinearIndex(i).observed) ++count;
    }
    py::array_t<double> result({py::ssize_t(count),py::ssize_t(4)});
    auto out=result.mutable_unchecked<2>(); size_t row=0;
    py::gil_scoped_release release;
    for (const auto& index : blocks) {
      const auto& block=esdf_layer_->getBlockByIndex(index);
      for(size_t i=0;i<block.num_voxels();++i) {
        const auto& voxel=block.getVoxelByLinearIndex(i);
        if(!voxel.observed) continue;
        const auto point=block.computeCoordinatesFromLinearIndex(i);
        for(int d=0;d<3;++d) out(row,d)=point(d);
        out(row,3)=voxel.distance; ++row;
      }
    }
    return result;
  }
 private:
  std::unique_ptr<voxblox::Layer<voxblox::TsdfVoxel>> layer_;
  std::unique_ptr<voxblox::TsdfIntegratorBase> integrator_;
  std::unique_ptr<voxblox::Layer<voxblox::EsdfVoxel>> esdf_layer_;
  std::unique_ptr<voxblox::EsdfIntegrator> esdf_integrator_;
};
PYBIND11_MODULE(_voxblox_native, m) {
  m.attr("upstream_revision")="c8066b04075d2fee509de295346b1c0b788c4f38";
  m.attr("upstream_patch")="parallel_ray_zero_guard";
  m.attr("api_version")=2;
  py::class_<TsdfVolume>(m,"TsdfVolume")
    .def(py::init<double,int,double,double,double,int>())
    .def("integrate",&TsdfVolume::integrate).def("query",&TsdfVolume::query).def("observed",&TsdfVolume::observed)
    .def("restore",&TsdfVolume::restore).def("crop",&TsdfVolume::crop)
    .def("configure_esdf",&TsdfVolume::configure_esdf)
    .def("update_esdf",&TsdfVolume::update_esdf).def("esdf_observed",&TsdfVolume::esdf_observed)
    .def_property_readonly("blocks",&TsdfVolume::blocks);
}
