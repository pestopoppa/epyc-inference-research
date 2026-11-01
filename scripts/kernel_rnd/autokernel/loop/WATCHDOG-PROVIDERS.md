# Restartable provider migration

A long-running standalone entry must be supervised. Passing an already initialized
registry to unified_driver.main with --listen, or a runtime factory to
campaign_service.main, refuses before the lane starts with HELD. Migrate the
application's provider assembly to an explicitly registered, no-argument
provider factory in provider_factories.py, and invoke unified_driver with
--provider-factory and its standalone startup manifest. The child constructs
the registry in its fresh interpreter; no objects are pickled or discovered.
The registry is intentionally empty until an application provides and pins a
real constructor. No production provider is currently installed by this change.
The non-listening CLI with no provider follows its existing path. The private
_main seam exists for in-process tests, not deployment.
