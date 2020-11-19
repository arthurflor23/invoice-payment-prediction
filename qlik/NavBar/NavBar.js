define(["qlik", "./properties", "text!./template.html", "css!./style.css"],
	function (qlik, qProp, template, style) {
		return {
			definition: qProp,
			template: template,
			style: style,
			support: {
				snapshot: true,
				export: true,
				exportData: false
			},
			controller: ['$scope', function ($scope) {
				$scope.initElements = async function (qKey) {
					await (async (k) => $scope.key = `${k}-${$scope.layout.qInfo.qId}`)(qKey);
					$('#qs-toolbar-container').append($(`#${$scope.key}-navbar`));
				}

				$scope.destroyElements = function () {
					$scope.component.model.Validated.unbind();

					angular.forEach($(`[id*='${$scope.key}']`), (elem) => $(elem).remove());
					angular.forEach($(`[class*='${$scope.key}']`), (elem) => {
						$(elem).removeClass((_, style) => {
							return style.split(' ').filter((w) => w.includes($scope.key)).join(' ');
						});
					});
				}

				$scope.changePath = function (path, option) {
					var dividor = '/state/';
					var index = window.location.pathname.indexOf(dividor);
					var pathname = window.location.pathname.substring(0, index + dividor.length);
					window.open(pathname + path, option);
				}

				$scope.updateLogoUrl = function () {
					setTimeout(() => {
						if (!$(`[tid='${$scope.layout.qInfo.qId}']`).hasClass('active')) return;

						var target = $("[tid='logoImg']").find('img');
						if (!target) return;

						target.attr("src", ($scope.layout.qProp.logoUrl && $scope.layout.qProp.logoUrl.startsWith("/")) ?
							window.location.origin + $scope.layout.qProp.logoUrl : $scope.layout.qProp.logoUrl);
					}, 100);
				};

				$(`[tid='qv-object-${$scope.layout.qInfo.qType}'] .qv-inner-object,
				.qv-panel-properties, .qv-properties button`).bind("click", () => {
					$scope.updateLogoUrl();
				});

				$scope.$on('$destroy', () => $scope.destroyElements());
				$scope.component.model.Validated.bind(() => $scope.updateLogoUrl());

				$scope.initElements(qProp.qKey);
				$scope.updateLogoUrl();

				$scope.html = "Hello World";
				console.log($scope.layout);
			}]
		};
	});